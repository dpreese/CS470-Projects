import javax.swing.*;
import java.awt.event.*;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.lang.*;
import javax.swing.JComponent;
import javax.swing.JFrame;
import java.io.*;
import java.util.Random;
import java.util.Scanner;
import java.net.*;

// This is the main class that you will add to in order to complete the lab
public class theRobot extends JFrame {
    // Mapping of actions to integers
    public static final int NORTH = 0;
    public static final int SOUTH = 1;
    public static final int EAST = 2;
    public static final int WEST = 3;
    public static final int STAY = 4;

    static mySmartMap myMaps; // instance of the class that draw everything to the GUI
    String mundoName;
    World mundo;
    double moveProb, sensorAccuracy;  // stores probabilies that the robot moves in the intended direction and the probability that a sonar reading is correct, respectively

    public Socket s;
    public BufferedReader sin;
    public PrintWriter sout;

    boolean isManual = false;
    boolean knownPosition = false;
    int startX = -1, startY = -1;
    int decisionDelay = 250;

    // store your probability map (for position of the robot in this array
    double[][] probs;

    // store your computed value of being in each state (x, y)
    double[][] Vs;

    public theRobot(String _manual, int _decisionDelay) {
        // initialize variables as specified from the command-line
        if (_manual.equals("automatic"))
            isManual = false;
        else
            isManual = true;
        decisionDelay = _decisionDelay;

        // get a connection to the server and get initial information about the world
        initClient();
        mundo = new World(mundoName);

        // set up the GUI that displays the information you compute
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
    //   2 -- it's transition model (specified by moveProb)
    //   3 -- it's sensor model (specified by sensorAccuracy)
    //   4 -- whether it's initial position is known.  if known, its position is stored in (startX, startY)
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
    // 'i' specifies the movement up
    // ',' specifies the movement down
    // 'l' specifies the movement right
    // 'j' specifies the movement left
    // 'k' specifies the movement stay
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
        // if the robot's initial position is known, reflect that in the probability map
        if (knownPosition) {
            for (int y = 0; y < mundo.height; y++) {
                for (int x = 0; x < mundo.width; x++) {
                    probs[x][y] = (x == startX && y == startY) ? 1.0 : 0.0;
                }
            }
        } else {  // otherwise, set up a uniform prior over all the positions in the world that are open spaces
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

    // TODO: update the probabilities of where the AI thinks it is based on the action selected and the new sonar readings
    //       To do this, you should update the 2D-array "probs"
    // Note: sonars is a bit string with four characters, specifying the sonar reading in the direction of North, South, East, and West
    //       For example, the sonar string 1001, specifies that the sonars found a wall in the North and West directions, but not in the South and East directions
    void updateProbabilities(int action, String sonars) {
        boolean[] sensors = new boolean[4];
        for (int i = 0; i < 4; i++) {
            sensors[i] = (sonars.charAt(i) == '1');
        }
        char actChar = actionToChar(action);
        int width = mundo.width, height = mundo.height;
        double[][] newProbs = new double[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (!isValid(x, y)) continue;
                double sum = 0;
                for (int px = 0; px < width; px++) {
                    for (int py = 0; py < height; py++) {
                        if (!isValid(px, py)) continue;
                        double transProb = getTransitionProbability(px, py, x, y, actChar);
                        sum += transProb * probs[px][py];
                    }
                }
                newProbs[x][y] = sum * getSensorProbability(sensors, x, y);
            }
        }

        double total = 0;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                total += newProbs[x][y];

        if (total > 0) {
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    newProbs[x][y] /= total;
        }

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                if (mundo.grid[x][y] == 2 || mundo.grid[x][y] == 3)
                    newProbs[x][y] = 0;

        total = 0;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                total += newProbs[x][y];

        if (total > 0) {
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    newProbs[x][y] /= total;
        }

        probs = newProbs;
        myMaps.updateProbs(probs);
    }

    // This is the function you'd need to write to make the robot move using your AI;
    // You do NOT need to write this function for this lab; it can remain as is
    int automaticAction() {
        return STAY;
    }

    void doStuff() {
        int action;

        //valueIteration();  // TODO: function you will write in Part II of the lab
        initializeProbabilities();  // Initializes the location (probability) map

        while (true) {
            try {
                action = isManual ? getHumanAction() : automaticAction(); // get the action selected by the user (from the keyboard) or AI
                sout.println(action); // send the action to the Server
                String sonars = sin.readLine(); // get sonar readings after the robot moves
                updateProbabilities(action, sonars); // TODO: this function should update the probabilities of where the AI thinks it is

                // check to see if the robot has reached its goal or fallen down stairs
                if (sonars.length() > 4) {
                    if (sonars.charAt(4) == 'w') {
                        myMaps.setWin();
                        break;
                    } else if (sonars.charAt(4) == 'l') {
                        myMaps.setLoss();
                        break;
                    }
                }
                Thread.sleep(decisionDelay);  // delay that is useful to see what is happening when the AI selects actions
            } catch (IOException | InterruptedException e) {
                System.out.println(e);
            }
        }
    }

    // java theRobot [manual/automatic] [delay]
    public static void main(String[] args) {
        new theRobot(args[0], Integer.parseInt(args[1]));
    }

    private char actionToChar(int action) {
        return switch (action) {
            case NORTH -> 'i';
            case SOUTH -> ',';
            case EAST -> 'l';
            case WEST -> 'j';
            default -> 'k';
        };
    }

    private boolean isValid(int x, int y) {
        return x >= 0 && y >= 0 && x < mundo.width && y < mundo.height && mundo.grid[x][y] != 1;
    }

    private int[] getDeltaFromAction(char action) {
        return switch (action) {
            case 'i' -> new int[]{0, -1};
            case ',' -> new int[]{0, 1};
            case 'j' -> new int[]{-1, 0};
            case 'l' -> new int[]{1, 0};
            case 'k' -> new int[]{0, 0};
            default -> new int[]{0, 0};
        };
    }

    private double getTransitionProbability(int fromX, int fromY, int toX, int toY, char action) {
        if (!isValid(toX, toY)) return 0;

        int[] intended = getDeltaFromAction(action);
        int intendedX = fromX + intended[0];
        int intendedY = fromY + intended[1];

        if (toX == fromX && toY == fromY) {
            double prob = (action == 'k' || !isValid(intendedX, intendedY)) ? moveProb : 0;
            int blocked = 0;
            for (char dir : new char[]{'i', ',', 'j', 'l'}) {
                int[] d = getDeltaFromAction(dir);
                int nx = fromX + d[0], ny = fromY + d[1];
                if (!isValid(nx, ny)) blocked++;
            }
            prob += (1 - moveProb) * blocked / 4.0;
            return prob;
        }

        if (toX == intendedX && toY == intendedY) return moveProb;

        for (char dir : new char[]{'i', ',', 'j', 'l'}) {
            int[] d = getDeltaFromAction(dir);
            if (fromX + d[0] == toX && fromY + d[1] == toY)
                return (1 - moveProb) / 4.0;
        }

        return 0;
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
