package dataMining;

import weka.classifiers.rules.JRip;
import weka.core.Instances;


public class ClassificatorUtils {

    public JRip teachClassifier(JRip jrip, Instances training, int parameter) {

        try {
            jrip.setSeed(parameter);
            jrip.buildClassifier(training);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return jrip;
    }

}
