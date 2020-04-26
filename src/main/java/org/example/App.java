package org.example;

import com.sun.org.apache.xpath.internal.operations.Mult;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class App
{
    MultiLayerNetwork network;
    DataSetIterator testIterator;
    UIServer uiServer;
    File trainData, testData;
    double learningRate = 0.001;
    double momentum = 0.9;
    double duration;

    public App() throws IOException
    {
        //configure training data and normalizer
        trainData = new File("D:\\cam29\\Downloads\\100-bird-species\\180\\train");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random());
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(224,224,3, labelMaker);
        rr.initialize(trainSplit);
        DataSetIterator trainIterator = new RecordReaderDataSetIterator(rr,64,1,180);
        DataNormalization imageScaler = new ImagePreProcessingScaler();
        imageScaler.fit(trainIterator);
        trainIterator.setPreProcessor(imageScaler);

        //configure testing data
        testData = new File("D:\\cam29\\Downloads\\100-bird-species\\180\\test");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random());
        ImageRecordReader rrTest = new ImageRecordReader(224,224,3, labelMaker);
        rrTest.initialize(testSplit);
        testIterator = new RecordReaderDataSetIterator(rrTest, 64, 1, 180);
        testIterator.setPreProcessor(imageScaler);

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
                //.l1(0.0001)
                .l2(0.0002)
                //.dropOut(0.8)
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

        //initialize network
        network = new MultiLayerNetwork(configuration);
        network.init();

        network.setListeners(new ScoreIterationListener(10), new TimeIterationListener(10));
        attachUI(network);

        //start network and evaluate once finished
        double start_time = System.currentTimeMillis();
        network.fit(trainIterator, 15);
        Evaluation evaluation = network.evaluate(testIterator);
        double duration = (System.currentTimeMillis() - start_time)/1000/60;

        //print results if run was successful
        if (evaluation.accuracy() > 0.15)
            printOutput(true, evaluation);

        //otherwise repeat
        else
        {
            System.out.println("Accuracy low at " + evaluation.accuracy());
            System.out.println("Repeating...");
            new App();
        }
        uiServer.stop();
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
            fileWriter.append(e.stats());
            fileWriter.append("This run took ").append(String.valueOf(duration)).append(" minutes.");
            fileWriter.append(network.evaluateROCMultiClass(testIterator, 0).stats());
        }
            System.out.println(e.stats());
            System.out.println("This run took " + duration + " minutes.");
            System.out.println(network.evaluateROCMultiClass(testIterator, 0));
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
