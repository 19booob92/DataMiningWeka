package main;

import java.io.BufferedReader;

import javax.swing.JOptionPane;

import utils.FileUtils;
import utils.States;
import weka.classifiers.rules.JRip;
import weka.core.Instance;
import weka.core.Instances;
import dataMining.ClassificatorUtils;
import dataMining.DataProcessor;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

public class Runner {

    private static ClassificatorUtils classificatorUtils = new ClassificatorUtils();
    private static DataProcessor dataProcessor = new DataProcessor();

    private static final int SEED = 1410;
    private static final int DIFF = 10;

    private static final double WEIGHT_IMPROVE_RATIO = 10.76;

    public static void main(String[] args) throws Exception {

        BufferedReader lines = FileUtils.loadFile(States.FILE_PATH);

        Instances instances = new Instances(lines);
        instances.setClassIndex(instances.numAttributes() - 1);

        Instances test = new Instances(instances, 0);
        Instances trainning = new Instances(instances, 0);

        BufferedReader fileToClassify = FileUtils.loadFile(States.FILE_TO_CLASSIFY_PATH);

        Instances instancesToClassify = new Instances(fileToClassify);
        instancesToClassify.setClassIndex(instancesToClassify.numAttributes() - 1);
        dataProcessor.splitInstances(trainning, test, instances);

        evaluate(test, trainning);

        JOptionPane.showMessageDialog(null, "zakończono");
    }

    public static void evaluate(Instances test, Instances training) throws Exception {
// SMOTE wykonuje się długo i nie przynosi rzadnych efektów
//        training = setFilter(test, training, 92);
        int progress = 0;
        final int SUM_OF_INSTANCES = (test.numInstances() + training.numInstances()) * ((int) (DIFF / 2));

        int truePositive = 0;
        int trueNegative = 0;
        int falsePositive = 0;
        int falseNegative = 0;

        double gini = 0.0;
        double currentGini = 0.0;

        double truePossitiveRate = 0.0;
        double trueNegativeRate = 0.0;

        JRip jRip = new JRip();

        for (int k = 0; k < training.numInstances(); k++) {
            Instance instance = training.instance(k);

            double classVal = instance.classValue();
            if (classVal == 1.0) {
                instance.setWeight(WEIGHT_IMPROVE_RATIO);
            }
            progress++;
        }
        System.err.println(progress + " / " + SUM_OF_INSTANCES + "     " + ((double) progress / SUM_OF_INSTANCES) * 100 + " %");

        jRip = classificatorUtils.teachClassifier(jRip, training, SEED);

// klasyfikuja albo nie
//            setClasses(jRip, test);
        for (int i = 0; i < test.numInstances(); i++) {
            double classification = jRip.classifyInstance(test.instance(i));

            if (classification == test.instance(i).classValue()) {
                if (classification == 1.0) {
                    truePositive++;
                } else {
                    trueNegative++;
                }
            } else {
                if (classification == 1.0) {
                    falsePositive++;
                } else {
                    falseNegative++;
                }
            }
            progress++;
        }
        System.err.println(progress + " / " + SUM_OF_INSTANCES + "     " + ((double) progress / SUM_OF_INSTANCES) * 100 + " %");

        trueNegativeRate = trueNegative * 100 / ((trueNegative + falsePositive) * 1.0);
        truePossitiveRate = truePositive * 100 / ((truePositive + falseNegative) * 1.0);

        currentGini = Math.sqrt(truePossitiveRate * trueNegativeRate);

        if (currentGini >= gini) {
            gini = currentGini;
        }

        System.err.println(
                "TruePossitive : " + truePositive);
        System.err.println(
                "FalsePossitive : " + falsePositive);
        System.err.println(
                "TrueNegative : " + trueNegative);
        System.err.println(
                "FalseNegative : " + falseNegative);

        System.err.println(
                " Gmean : " + gini);

        classifyAndSave(jRip, test);
    }

    private static Instances setFilter(Instances test, Instances training, int percentage) throws Exception {
        SMOTE smote = new SMOTE();
        smote.setPercentage(percentage);
        smote.setInputFormat(training);

        return Filter.useFilter(training, smote);
    }

    public static void classifyAndSave(JRip classifier, Instances test) throws Exception {
        FileUtils.saveLabeledInstanced(test);
    }

    private static void setClasses(JRip classifier, Instances test) throws Exception {
        for (int i = 0; i < test.numInstances(); i++) {
            Instance instance = test.instance(i);

            double classyfication = classifier.classifyInstance(instance);

            instance.setClassValue(classyfication);
        }
    }
}
