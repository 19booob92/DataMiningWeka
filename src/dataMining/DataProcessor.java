package dataMining;

import java.util.Random;

import weka.core.Instances;


public class DataProcessor {

    public void splitInstances(Instances trainning, Instances test, Instances instances) {
        Random rand = new Random();

        for (int i = 0; i < instances.numInstances(); i++) {
            int value = rand.nextInt(100);
            if (value <= 50) {
                trainning.add(instances.instance(i));
            } else {
                test.add(instances.instance(i));
            }
        }
    }
}
