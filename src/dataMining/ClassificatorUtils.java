package dataMining;

import weka.classifiers.rules.JRip;
import weka.core.Instances;


public class ClassificatorUtils {

    public JRip teachClassifier(JRip jrip, Instances training, int parameter) {

        JRip jRip = jrip;
        try {
            jRip.setSeed(parameter);
            jRip.buildClassifier(training);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return jRip;
    }

}
