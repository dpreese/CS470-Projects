import javax.swing.*;
import java.awt.event.*;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.io.*;
import java.net.*;
import java.util.Random;

// This is the main class that you will add to in order to complete the lab
public class theRobot extends JFrame {
    // Mapping of actions to integers
    public static final int NORTH = 0;
    public static final int SOUTH = 1;
    public static final int EAST  = 2;
    public static final int WEST  = 3;
    public static final int STAY  = 4;

    static mySmartMap myMaps; // instance of the class that draws everything to the GUI
    String mundoName;
    World mundo;
    double moveProb, sensorAccuracy;  // stores probabilities for motion and sonar accuracy

    public Socket s;
    public BufferedReader sin;
    public PrintWriter sout;

    boolean isManual = false;
    boolean knownPosition = false;
    int startX = -1, startY = -1;
    int decisionDelay = 250;

    // store your probability map (for robot position)
    double[][] probs;

    // store your computed value of being in each state (x, y)
    double[][] Vs;

    // Random for exploration
    private static final double EXPLORATION_RATE = 0.1;  // probability to explore randomly
    private final Random rand = new Random();

    // Value iteration parameters
    private static final double GAMMA = 0.9;
    private static final double EPSILON = 1e-3;

    public theRobot(String _manual, int _decisionDelay) {
        // initialize variables as specified from the command-line
        isManual = !_manual.equals("automatic");
        decisionDelay = _decisionDelay;

        // get a connection to the server and get initial information about the world
        initClient();
        mundo = new World(mundoName);

        // set up the GUI
        int width = 500;
        int height = 500;
        int bar = 20;
        setSize(width, height + bar);
        getContentPane().setBackground(new Color(230, 230, 230));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setBounds(0, 0, width, height + bar);
        myMaps = new mySmartMap(width, height, mundo);
        getContentPane().add(myMaps);
        setVisible(true);
        setTitle("Probability and Value Maps");

        doStuff();
    }

    // this function establishes a connection with the server and learns
    //   1 -- which world it is in
    //   2 -- its transition model (moveProb)
    //   3 -- its sensor model (sensorAccuracy)
    //   4 -- whether its initial position is known
    public void initClient() {
        int portNumber = 3333;
        String host = "localhost";
        try {
            s = new Socket(host, portNumber);
            sout = new PrintWriter(s.getOutputStream(), true);
            sin = new BufferedReader(new InputStreamReader(s.getInputStream()));

            mundoName = sin.readLine();
            moveProb = Double.parseDouble(sin.readLine());
            sensorAccuracy = Double.parseDouble(sin.readLine());

            String _known = sin.readLine();
            if (_known.equals("known")) {
                knownPosition = true;
                startX = Integer.parseInt(sin.readLine());
                startY = Integer.parseInt(sin.readLine());
            }
        } catch (IOException e) {
            System.err.println("Caught IOException: " + e.getMessage());
        }
    }

    // function that gets human-specified actions
    int getHumanAction() {
        while (myMaps.currentKey < 0) {
            try {
                Thread.sleep(50);
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
        int a = myMaps.currentKey;
        myMaps.currentKey = -1;
        return a;
    }

    // initializes the probabilities of where the AI is
    void initializeProbabilities() {
        probs = new double[mundo.width][mundo.height];
        if (knownPosition) {
            for (int y = 0; y < mundo.height; y++)
                for (int x = 0; x < mundo.width; x++)
                    probs[x][y] = (x == startX && y == startY) ? 1.0 : 0.0;
        } else {
            int count = 0;
            for (int y = 0; y < mundo.height; y++)
                for (int x = 0; x < mundo.width; x++)
                    if (mundo.grid[x][y] == 0) count++;
            for (int y = 0; y < mundo.height; y++)
                for (int x = 0; x < mundo.width; x++)
                    probs[x][y] = (mundo.grid[x][y] == 0) ? 1.0 / count : 0;
        }
        myMaps.updateProbs(probs);
    }

    // Bayesian filter update
    void updateProbabilities(int action, String sonars) {
        boolean[] sensors = new boolean[4];
        for (int i = 0; i < 4; i++) sensors[i] = (sonars.charAt(i) == '1');
        int width = mundo.width, height = mundo.height;
        double[][] newProbs = new double[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (!isValid(x, y)) continue;
                double sum = 0;
                for (int px = 0; px < width; px++) {
                    for (int py = 0; py < height; py++) {
                        if (!isValid(px, py)) continue;
                        sum += getTransitionProbability(px, py, x, y, actionToChar(action)) * probs[px][py];
                    }
                }
                newProbs[x][y] = sum * getSensorProbability(sensors, x, y);
            }
        }
        // normalize
        double total = 0;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) total += newProbs[x][y];
        if (total > 0) {
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++) newProbs[x][y] /= total;
        }
        // zero out terminal squares
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                if (mundo.grid[x][y] == 2 || mundo.grid[x][y] == 3)
                    newProbs[x][y] = 0;
        // renormalize
        total = 0;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) total += newProbs[x][y];
        if (total > 0) {
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++) newProbs[x][y] /= total;
        }
        probs = newProbs;
        myMaps.updateProbs(probs);
    }

    //------------------------- PART II: VALUE ITERATION -------------------------
    // Reward function
    // Reward function
    private double reward(int x, int y) {
        switch (mundo.grid[x][y]) {
            case 2: return 100;   // goal
            case 3: return -100;  // stairwell/fall
            default: return -1;   // open space
        }
    }

    // Value Iteration algorithm
    void valueIteration() {
        int w = mundo.width, h = mundo.height;
        Vs = new double[w][h];
        double[][] newV = new double[w][h];
        boolean converged;

        do {
            converged = true;
            for (int x = 0; x < w; x++) {
                for (int y = 0; y < h; y++) {
                    if (!isValid(x, y)) continue;
                    double maxQ = Double.NEGATIVE_INFINITY;
                    for (int a = NORTH; a <= STAY; a++) {
                        double q = 0;
                        char actChar = actionToChar(a);
                        for (int nx = 0; nx < w; nx++) {
                            for (int ny = 0; ny < h; ny++) {
                                if (!isValid(nx, ny)) continue;
                                q += getTransitionProbability(x, y, nx, ny, actChar) * Vs[nx][ny];
                            }
                        }
                        q = reward(x, y) + GAMMA * q;
                        maxQ = Math.max(maxQ, q);
                    }
                    newV[x][y] = maxQ;
                    if (Math.abs(newV[x][y] - Vs[x][y]) > EPSILON) converged = false;
                }
            }
            for (int x = 0; x < w; x++) System.arraycopy(newV[x], 0, Vs[x], 0, h);
        } while (!converged);

        myMaps.updateValues(Vs);
    }

    // Automatic action via expected utility with exploration
    int automaticAction() {
        // epsilon-greedy: explore randomly some of the time
        if (rand.nextDouble() < EXPLORATION_RATE) {
            // pick a random action excluding STAY to encourage movement
            int[] actions = new int[]{NORTH, SOUTH, EAST, WEST};
            return actions[rand.nextInt(actions.length)];
        }

        int w = mundo.width, h = mundo.height;
        double bestEV = Double.NEGATIVE_INFINITY;
        int bestAction = STAY;

        for (int a = NORTH; a <= STAY; a++) {
            double ev = 0;
            char actChar = actionToChar(a);
            for (int x = 0; x < w; x++) {
                for (int y = 0; y < h; y++) {
                    if (!isValid(x, y)) continue;
                    double q = 0;
                    for (int nx = 0; nx < w; nx++) {
                        for (int ny = 0; ny < h; ny++) {
                            if (!isValid(nx, ny)) continue;
                            q += getTransitionProbability(x, y, nx, ny, actChar) * Vs[nx][ny];
                        }
                    }
                    ev += probs[x][y] * q;
                }
            }
            if (ev > bestEV) {
                bestEV = ev;
                bestAction = a;
            }
        }
        return bestAction;
    }

    void doStuff() {
        int action;

        valueIteration();       // compute value map before starting
        initializeProbabilities();  // Initializes the probability map

        while (true) {
            try {
                action = isManual ? getHumanAction() : automaticAction();
                sout.println(action);
                String sonars = sin.readLine();
                updateProbabilities(action, sonars);

                if (sonars.length() > 4) {
                    if (sonars.charAt(4) == 'w') {
                        myMaps.setWin();
                        break;
                    } else if (sonars.charAt(4) == 'l') {
                        myMaps.setLoss();
                        break;
                    }
                }
                Thread.sleep(decisionDelay);
            } catch (IOException | InterruptedException e) {
                System.out.println(e);
            }
        }
    }

    public static void main(String[] args) {
        new theRobot(args[0], Integer.parseInt(args[1]));
    }

    // Helper methods
    private char actionToChar(int action) {
        return switch (action) {
            case NORTH -> 'i';
            case SOUTH -> ',';
            case EAST  -> 'l';
            case WEST  -> 'j';
            default    -> 'k';
        };
    }

    private boolean isValid(int x, int y) {
        return x >= 0 && y >= 0 && x < mundo.width && y < mundo.height && mundo.grid[x][y] != 1;
    }

    private double getTransitionProbability(int fromX, int fromY, int toX, int toY, char action) {
        if (!isValid(toX, toY)) return 0;
        int[] delta = getDeltaFromAction(action);
        int intendedX = fromX + delta[0], intendedY = fromY + delta[1];

        if (toX == intendedX && toY == intendedY) return moveProb;
        if (toX == fromX && toY == fromY) {
            double stayProb = (action == 'k' || !isValid(intendedX, intendedY)) ? moveProb : 0;
            int blocked = 0;
            for (char dir : new char[]{'i', ',', 'j', 'l'}) {
                int[] d = getDeltaFromAction(dir);
                if (!isValid(fromX + d[0], fromY + d[1])) blocked++;
            }
            stayProb += (1 - moveProb) * blocked / 4.0;
            return stayProb;
        }
        // unintended moves
        for (char dir : new char[]{'i', ',', 'j', 'l'}) {
            int[] d = getDeltaFromAction(dir);
            if (fromX + d[0] == toX && fromY + d[1] == toY) return (1 - moveProb) / 4.0;
        }
        return 0;
    }

    private int[] getDeltaFromAction(char action) {
        return switch (action) {
            case 'i' -> new int[]{0, -1};
            case ',' -> new int[]{0, 1};
            case 'j' -> new int[]{-1, 0};
            case 'l' -> new int[]{1, 0};
            default  -> new int[]{0, 0};
        };
    }

    private double getSensorProbability(boolean[] reading, int x, int y) {
        if (!isValid(x, y)) return 0;
        boolean[] actual = new boolean[4];
        actual[0] = !isValid(x, y - 1);
        actual[1] = !isValid(x, y + 1);
        actual[2] = !isValid(x - 1, y);
        actual[3] = !isValid(x + 1, y);

        double prob = 1.0;
        for (int i = 0; i < 4; i++) {
            prob *= (reading[i] == actual[i]) ? sensorAccuracy : (1 - sensorAccuracy);
        }
        return prob;
    }
}
