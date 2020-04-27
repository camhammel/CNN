package org.example;

import com.sun.org.apache.xpath.internal.operations.Mult;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.ui.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class App
{
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(App.class);
    MultiLayerNetwork network;
    DataSetIterator trainIterator, testIterator, validIterator;
    UIServer uiServer;
    File trainData, testData, validData;
    double learningRate = 0.001;
    double momentum = 0.9;
    double duration;

    public App() throws IOException
    {
        //configure training data and normalizer
        trainData = new File("C:\\Users\\Cameron\\IdeaProjects\\CNN\\data\\inputs\\100-bird-species\\180\\train");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random());
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(224,224,3, labelMaker);
        rr.initialize(trainSplit);
        trainIterator = new RecordReaderDataSetIterator(rr,64,1,180);
        DataNormalization imageScaler = new ImagePreProcessingScaler();
        imageScaler.fit(trainIterator);
        trainIterator.setPreProcessor(imageScaler);

        //configure testing data
        testData = new File("C:\\Users\\Cameron\\IdeaProjects\\CNN\\data\\inputs\\100-bird-species\\180\\test");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random());
        ImageRecordReader rrTest = new ImageRecordReader(224,224,3, labelMaker);
        rrTest.initialize(testSplit);
        testIterator = new RecordReaderDataSetIterator(rrTest, 64, 1, 180);
        testIterator.setPreProcessor(imageScaler);

        //configure validation data
        validData = new File("C:\\Users\\Cameron\\IdeaProjects\\CNN\\data\\inputs\\100-bird-species\\180\\valid");
        FileSplit validSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random());
        ImageRecordReader rrValid = new ImageRecordReader(224,224,3, labelMaker);
        rrValid.initialize(validSplit);
        validIterator = new RecordReaderDataSetIterator(rrValid, 64, 1, 180);
        validIterator.setPreProcessor(imageScaler);

        //configure network structure
        ConvolutionLayer layer0 = new ConvolutionLayer.Builder(5,5)
                .nIn(3)
                .nOut(16)
                .stride(1,1)
                .padding(2,2)
                .weightInit(WeightInit.XAVIER)
                .name("First convolution layer")
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .name("First subsampling layer")
                .build();

        ConvolutionLayer layer2 = new ConvolutionLayer.Builder(5,5)
                .nOut(20)
                .stride(1,1)
                .padding(2,2)
                .weightInit(WeightInit.XAVIER)
                .name("Second convolution layer")
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .name("Second subsampling layer")
                .build();

        ConvolutionLayer layer4 = new ConvolutionLayer.Builder(5,5)
                .nOut(20)
                .stride(1,1)
                .padding(2,2)
                .weightInit(WeightInit.XAVIER)
                .name("Third convolution layer")
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .name("Third subsampling layer")
                .build();

        OutputLayer layer6 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .name("Output")
                .nOut(180)
                .build();

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(System.currentTimeMillis())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(0.0001)
                .l2(0.0001)
                .dropOut(0.8)
                .updater(new Nesterovs(learningRate, momentum))
                .list()
                .layer(0, layer0)
                .layer(1, layer1)
                .layer(2, layer2)
                .layer(3, layer3)
                .layer(4, layer4)
                .layer(5, layer5)
                .layer(6, layer6)
                .setInputType(InputType.convolutional(224,224,3))
                .build();

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(30))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(60, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(validIterator, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("C:\\Users\\Cameron\\IdeaProjects\\CNN\\data\\outputs\\models"))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, configuration, trainIterator);
        EarlyStoppingResult result = trainer.fit();

        System.out.println("Terminated due to: " + result.getTerminationReason());
        System.out.println(result.getTerminationDetails());
        System.out.println("Best epoch: " + result.getBestModelEpoch() + " / " + result.getTotalEpochs() + " total epochs.");
        System.out.println("Score at best epoch: " + result.getBestModelScore());
        //Model model  = result.getBestModel();

        //initialize network
        //network = new MultiLayerNetwork(configuration);
        //network.init();
        //log.info(network.summary());
        //network.setListeners(new ScoreIterationListener(10), new TimeIterationListener(10));
        //attachUI(network);

        //start network and evaluate once finished
        //double start_time = System.currentTimeMillis();
        //network.fit(trainIterator, 25);
        //System.out.println("Training complete. Testing...");
        //Evaluation evaluation = network.evaluate(testIterator);
        //duration = (System.currentTimeMillis() - start_time)/1000/60;

        //print results if run was successful
        //if (evaluation.accuracy() > 0.15)
        //    printOutput(true, evaluation);

        //otherwise repeat
        //else
        //{
        //    System.out.println("Accuracy low at " + evaluation.accuracy());
        //    System.out.println("Repeating...");
        //    new App();
        //}
        //uiServer.stop();
    }

    public void printOutput(boolean toFile, Evaluation e) throws IOException
    {
        if (toFile)
        {
            FileWriter fileWriter;
            try {
                fileWriter = new FileWriter("C:\\Users\\Cameron\\IdeaProjects\\CNN\\data\\outputs\\" + learningRate + "," + momentum + ".txt");
            }
            catch (IOException ioException) {
                fileWriter = new FileWriter("C:\\Users\\Cameron\\IdeaProjects\\CNN\\data\\outputs\\" + learningRate + "," + momentum + "_1" + ".txt");
            }
            fileWriter.write(e.stats(true));
            fileWriter.write("\n" + e.confusionMatrix() + "\n");
            fileWriter.append("This run took ").append(String.valueOf(duration)).append(" minutes.").append("\n");
            fileWriter.append(network.evaluateROCMultiClass(testIterator, 0).stats());
            fileWriter.flush();
            fileWriter.close();
        }
            System.out.println(e.stats());
            System.out.println("This run took " + duration + " minutes.");
            System.out.println(network.evaluateROCMultiClass(testIterator, 0).stats());
    }

    public static void main(String[] args) {
        try {
            new App();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void attachUI(MultiLayerNetwork mln)
    {
        uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.detach(statsStorage);
        int listenerFrequency = 50;
        uiServer.attach(statsStorage);
        mln.setListeners(new StatsListener(statsStorage, listenerFrequency));
    }
}
