package com.plantdisease.classification;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.RBFClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.supervised.instance.ClassBalancer;

import java.io.File;
import java.util.Random;

public class PlantDiseaseClassifier {
    private Instances data;
    private Instances train;
    private Instances test;
    private Classifier classifier;
    private Evaluation evaluation;

    /**
     * Carrega o conjunto de dados (suporta CSV ou ARFF).
     * Após carregar, aplica normalização e balanceamento.
     *
     * @param filePath caminho para o arquivo de dados.
     * @throws Exception se ocorrer erro na leitura.
     */
    public void loadData(String filePath) throws Exception {
        if (filePath.toLowerCase().endsWith(".csv")) {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filePath));
            data = loader.getDataSet();
        } else if (filePath.toLowerCase().endsWith(".arff")) {
            DataSource source = new DataSource(filePath);
            data = source.getDataSet();
        } else {
            throw new IllegalArgumentException("Formato não suportado. Use CSV ou ARFF.");
        }
        // Define o atributo de classe (último atributo) se ainda não estiver definido.
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        // Aplica normalização e balanceamento
        preprocessData();
    }

    /**
     * Aplica normalização e balanceamento à base de dados.
     *
     * @throws Exception se ocorrer erro.
     */
    public void preprocessData() throws Exception {
        // Normalização
        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        Instances normalizedData = Filter.useFilter(data, normalize);

        // Balanceamento com ClassBalancer (gera dados sintéticos para equidade)
        ClassBalancer balancer = new ClassBalancer();
        balancer.setInputFormat(normalizedData);
        Instances balancedData = Filter.useFilter(normalizedData, balancer);

        data = balancedData;
    }

    /**
     * Retorna o conjunto de dados.
     *
     * @return Instances.
     */
    public Instances getData() {
        return data;
    }

    /**
     * Realiza a divisão Train/Test.
     *
     * @param trainPercentage porcentagem para treinamento (ex: 70 para 70%).
     */
    public void trainTestSplit(double trainPercentage) {
        data.randomize(new Random(1));
        int trainSize = (int) Math.round(data.numInstances() * trainPercentage / 100);
        int testSize = data.numInstances() - trainSize;
        train = new Instances(data, 0, trainSize);
        test = new Instances(data, trainSize, testSize);
    }

    /**
     * Configura o classificador com base na opção selecionada e parâmetros de tuning.
     *
     * Opções de modelos:
     *  1 – SVM (SMO): tuningParams[0] = C, tuningParams[1] = Expoente do Kernel.
     *  2 – Árvore de Decisão (J48): tuningParams[0] = Fator de Confiança, tuningParams[1] = Min Num Objetos.
     *  3 – Boosting (AdaBoostM1): tuningParams[0] = Número de Iterações.
     *  4 – RandomForest: tuningParams[0] = Número de Árvores.
     *  5 – IBk (k-NN): tuningParams[0] = k.
     *  6 – Rede Neural RBF (RBFClassifier): tuningParams[0] = Número de neurônios ocultos (ex.: 10).
     *
     * @param modelOption opção do modelo (1 a 6).
     * @param tuningParams vetor com os parâmetros de tuning.
     * @throws Exception se opção inválida.
     */
    public void setClassifier(int modelOption, double[] tuningParams) throws Exception {
        switch (modelOption) {
            case 1: // SVM (SMO)
                SMO smo = new SMO();
                smo.setC(tuningParams[0]);
                PolyKernel pk = new PolyKernel();
                pk.setExponent(tuningParams[1]);
                smo.setKernel(pk);
                classifier = smo;
                break;
            case 2: // Árvore de Decisão (J48)
                J48 tree = new J48();
                tree.setConfidenceFactor((float) tuningParams[0]);
                tree.setMinNumObj((int) tuningParams[1]);
                classifier = tree;
                break;
            case 3: // Boosting (AdaBoostM1)
                AdaBoostM1 boost = new AdaBoostM1();
                boost.setNumIterations((int) tuningParams[0]);
                classifier = boost;
                break;
            case 4: // RandomForest
                RandomForest rf = new RandomForest();
//                rf.setNumTrees((int) tuningParams[0]);
                classifier = rf;
                break;
            case 5: // IBk (k-NN)
                IBk ibk = new IBk();
                ibk.setKNN((int) tuningParams[0]);
                classifier = ibk;
                break;
            case 6: // Rede Neural RBF (RBFClassifier)
                RBFClassifier rbf = new RBFClassifier();
                // Define o número de neurônios ocultos (assumindo que o número de clusters corresponde aos neurônios ocultos)
                rbf.setNumFunctions((int) tuningParams[0]);
                rbf.setSeed((int) tuningParams[1]);
                boolean useCGD;
                if (tuningParams[2] >= 1){
                    useCGD = true;
                } else{
                    useCGD = false;
                }
                rbf.setUseCGD(useCGD);
                classifier = rbf;
                break;
            default:
                throw new IllegalArgumentException("Opção de modelo inválida. Use 1 a 6.");
        }
    }

    /**
     * Treina o classificador utilizando o conjunto de treinamento.
     *
     * @throws Exception se ocorrer erro.
     */
    public void trainClassifier() throws Exception {
        if (classifier == null) {
            throw new IllegalStateException("Classificador não configurado. Chame setClassifier() primeiro.");
        }
        if (train == null) {
            throw new IllegalStateException("Dados de treinamento não preparados. Chame trainTestSplit() se for usar Train/Test.");
        }
        classifier.buildClassifier(train);
        // Imprime os detalhes do modelo treinado (para portfólio)
        System.out.println("Modelo Treinado:\n" + classifier.toString());
    }

    /**
     * Avalia o classificador treinado utilizando o conjunto de teste.
     *
     * @throws Exception se ocorrer erro.
     */
    public void evaluateModel() throws Exception {
        if (classifier == null) {
            throw new IllegalStateException("Classificador não configurado ou não treinado.");
        }
        evaluation = new Evaluation(train);
        evaluation.evaluateModel(classifier, test);
    }

    /**
     * Avalia o classificador utilizando validação cruzada.
     *
     * @param folds número de folds.
     * @throws Exception se ocorrer erro.
     */
    public void evaluateModelCV(int folds) throws Exception {
        if (classifier == null) {
            throw new IllegalStateException("Classificador não configurado ou não treinado.");
        }
        evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, folds, new Random(1));
    }

    /**
     * Executa avaliação para múltiplos valores de k e gera uma tabela com as métricas:
     * Taxa de acerto (accuracy), precisão (média) e sensibilidade (recall médio).
     * Destaca o melhor, o caso médio e o pior.
     *
     * @param foldsArray array com os diferentes valores de k (ex: {5,10,15,20,25}).
     * @return Tabela em formato de string.
     * @throws Exception se ocorrer erro.
     */
    public String evaluateMultipleCV(int[] foldsArray) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("k,Taxa de Acerto,Precisão,Sensibilidade\n");
        double bestAcc = -1;
        double worstAcc = 101;
        int bestK = -1;
        int worstK = -1;
        double totalAcc = 0;
        int count = 0;
        for (int k : foldsArray) {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, k, new Random(1));
            double acc = eval.pctCorrect();
            double avgPrec = 0;
            double avgRec = 0;
            for (int i = 0; i < data.numClasses(); i++) {
                avgPrec += eval.precision(i);
                avgRec += eval.recall(i);
            }
            avgPrec /= data.numClasses();
            avgRec /= data.numClasses();
            sb.append(k).append(",")
                    .append(String.format("%.2f", acc)).append(",")
                    .append(String.format("%.2f", avgPrec)).append(",")
                    .append(String.format("%.2f", avgRec)).append("\n");
            if (acc > bestAcc) {
                bestAcc = acc;
                bestK = k;
            }
            if (acc < worstAcc) {
                worstAcc = acc;
                worstK = k;
            }
            totalAcc += acc;
            count++;
        }
        double avgAcc = totalAcc / count;
        sb.append("\nMelhor Caso: k = ").append(bestK)
                .append(" com Taxa de Acerto = ").append(String.format("%.2f", bestAcc));
        sb.append("\nPior Caso: k = ").append(worstK)
                .append(" com Taxa de Acerto = ").append(String.format("%.2f", worstAcc));
        sb.append("\nCaso Médio: Taxa de Acerto = ").append(String.format("%.2f", avgAcc));
        return sb.toString();
    }

    /**
     * Retorna o objeto Evaluation com os resultados.
     *
     * @return Evaluation.
     */
    public Evaluation getEvaluation() {
        return evaluation;
    }

    /**
     * Classifica uma instância.
     *
     * @param instance instância a ser classificada.
     * @return índice da classe prevista.
     * @throws Exception se ocorrer erro.
     */
    public double classifyInstance(Instance instance) throws Exception {
        if (classifier == null) {
            throw new IllegalStateException("Classificador não configurado ou não treinado.");
        }
        return classifier.classifyInstance(instance);
    }
}
