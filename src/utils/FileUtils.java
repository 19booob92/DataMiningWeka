package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import javax.swing.JOptionPane;

import weka.core.Instances;


public class FileUtils {

    public static BufferedReader loadFile(String fileName) {

        try {
            FileReader file = new FileReader(fileName);
            BufferedReader bufferedReader = new BufferedReader(file);
            return bufferedReader;
        } catch (FileNotFoundException e) {
            JOptionPane.showMessageDialog(null, "Brak pliku w odpowiednim katalogu");
        }

        return null;
    }

    public static void saveLabeledInstanced(Instances labeled) throws IOException {
        BufferedWriter writer = new BufferedWriter(
                new FileWriter(States.OUTPUT_FILE_PATH));
        writer.write(labeled.toString());
        writer.newLine();
        writer.flush();
        writer.close();
        
        JOptionPane.showMessageDialog(null, "Zapisano");
    }

}
