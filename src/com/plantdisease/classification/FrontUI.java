package com.plantdisease.classification;

import weka.classifiers.Evaluation;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class FrontUI extends JFrame {
    private JComboBox<String> modelComboBox;
    private JRadioButton trainTestRadio;
    private JRadioButton crossValRadio;
    private JTextField trainPercentageField;
    private JTextField foldsField;
    private JButton trainEvaluateButton;
    private JButton showModelSummaryButton;
    private JButton evalMultipleCVButton;
    private JTextArea evaluationTextArea;

    // Painel de tuning e campos
    private JPanel tuningPanel;
    private CardLayout tuningCardLayout;
    // SVM
    private JTextField svmCField;
    private JTextField svmKernelExpField;
    // J48
    private JTextField j48ConfidenceField;
    private JTextField j48MinNumField;
    // AdaBoost
    private JTextField adaBoostIterationsField;
    // RandomForest
    private JTextField rfNumTreesField;
    // IBk
    private JTextField ibkKField;
    // RBF (Rede Neural)
    private JTextField rbfHiddenField; // Número de neurônios ocultos (ex.: 10)
    private JTextField rbfSeedField; // Semente para neuronio.
    private JTextField rbfUseCGDFied; // Usar CGD.

    private PlantDiseaseClassifier classifier;

    public FrontUI() {
        super("Portfólio: Classificação de Doenças de Plantas");
        classifier = new PlantDiseaseClassifier();
        initUI();
    }

    private void initUI() {
        setLayout(new BorderLayout());

        // Painel de configuração
        JPanel configPanel = new JPanel(new BorderLayout());
        configPanel.setBorder(BorderFactory.createTitledBorder("Configuração"));

        // Parte superior: seleção de modelo e método de avaliação
        JPanel upperPanel = new JPanel(new GridLayout(2, 1));

        // Seleção de modelo
        JPanel modelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        modelPanel.add(new JLabel("Selecione o Modelo:"));
        String[] models = {"SVM (SMO)", "Árvore (J48)", "Boosting (AdaBoostM1)", "RandomForest", "IBk (k-NN)", "Rede Neural RBF"};
        modelComboBox = new JComboBox<>(models);
        modelPanel.add(modelComboBox);
        upperPanel.add(modelPanel);

        // Seleção do método de avaliação
        JPanel evalPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        evalPanel.add(new JLabel("Método de Avaliação:"));
        trainTestRadio = new JRadioButton("Train/Test Split", true);
        crossValRadio = new JRadioButton("Validação Cruzada (CV)");
        ButtonGroup evalGroup = new ButtonGroup();
        evalGroup.add(trainTestRadio);
        evalGroup.add(crossValRadio);
        evalPanel.add(trainTestRadio);
        evalPanel.add(crossValRadio);
        upperPanel.add(evalPanel);

        configPanel.add(upperPanel, BorderLayout.NORTH);

        // Parâmetros da avaliação
        JPanel evalParamPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        evalParamPanel.add(new JLabel("Treinamento % (Train/Test):"));
        trainPercentageField = new JTextField("70", 5);
        evalParamPanel.add(trainPercentageField);
        evalParamPanel.add(new JLabel("Folds (CV):"));
        foldsField = new JTextField("10", 5);
        evalParamPanel.add(foldsField);
        configPanel.add(evalParamPanel, BorderLayout.CENTER);

        // Painel de tuning (CardLayout)
        tuningPanel = new JPanel();
        tuningCardLayout = new CardLayout();
        tuningPanel.setLayout(tuningCardLayout);
        tuningPanel.setBorder(BorderFactory.createTitledBorder("Parâmetros de Tuning"));

        // SVM
        JPanel svmPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        svmPanel.add(new JLabel("C:"));
        svmCField = new JTextField("1.0", 5);
        svmPanel.add(svmCField);
        svmPanel.add(new JLabel("Expoente do Kernel:"));
        svmKernelExpField = new JTextField("1.0", 5);
        svmPanel.add(svmKernelExpField);
        tuningPanel.add(svmPanel, "SVM");

        // J48
        JPanel j48Panel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        j48Panel.add(new JLabel("Fator de Confiança:"));
        j48ConfidenceField = new JTextField("0.25", 5);
        j48Panel.add(j48ConfidenceField);
        j48Panel.add(new JLabel("Min Num Objetos:"));
        j48MinNumField = new JTextField("2", 5);
        j48Panel.add(j48MinNumField);
        tuningPanel.add(j48Panel, "J48");

        // AdaBoost
        JPanel adaBoostPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        adaBoostPanel.add(new JLabel("Número de Iterações:"));
        adaBoostIterationsField = new JTextField("10", 5);
        adaBoostPanel.add(adaBoostIterationsField);
        tuningPanel.add(adaBoostPanel, "AdaBoost");

        // RandomForest
        JPanel rfPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        rfPanel.add(new JLabel("Número de Árvores:"));
        rfNumTreesField = new JTextField("100", 5);
        rfPanel.add(rfNumTreesField);
        tuningPanel.add(rfPanel, "RandomForest");

        // IBk
        JPanel ibkPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        ibkPanel.add(new JLabel("k:"));
        ibkKField = new JTextField("1", 5);
        ibkPanel.add(ibkKField);
        tuningPanel.add(ibkPanel, "IBk");

        // Rede Neural RBF
        JPanel rbfPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        rbfPanel.add(new JLabel("Neurônios Ocultos:"));
        rbfHiddenField = new JTextField("10", 5);
        rbfPanel.add(rbfHiddenField);

        rbfPanel.add(new JLabel("Seed:"));
        rbfSeedField = new JTextField("42", 42);
        rbfPanel.add(rbfSeedField);

        rbfPanel.add(new JLabel("Usar CGD? 1 | 0"));
        rbfUseCGDFied = new JTextField("1", 1);
        rbfPanel.add(rbfUseCGDFied);

        tuningPanel.add(rbfPanel, "RBF");

        configPanel.add(tuningPanel, BorderLayout.SOUTH);
        add(configPanel, BorderLayout.NORTH);

        // Área de texto para métricas de avaliação
        evaluationTextArea = new JTextArea();
        evaluationTextArea.setEditable(false);
        evaluationTextArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        JScrollPane scrollPane = new JScrollPane(evaluationTextArea);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Métricas de Avaliação"));
        add(scrollPane, BorderLayout.CENTER);

        // Painel inferior com botões
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        trainEvaluateButton = new JButton("Train & Evaluate");
        showModelSummaryButton = new JButton("Show Model Summary");
        evalMultipleCVButton = new JButton("Evaluate Multiple CV");
        buttonPanel.add(trainEvaluateButton);
        buttonPanel.add(showModelSummaryButton);
        buttonPanel.add(evalMultipleCVButton);
        add(buttonPanel, BorderLayout.SOUTH);

        // Atualiza painel de tuning conforme modelo selecionado
        modelComboBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                updateTuningPanel();
            }
        });
        updateTuningPanel();

        // Ações dos botões
        trainEvaluateButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trainAndEvaluate();
            }
        });
        showModelSummaryButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                showModelSummary();
            }
        });
        evalMultipleCVButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                evaluateMultipleCVAction();
            }
        });

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(850, 650);
        setLocationRelativeTo(null);
    }

    private void updateTuningPanel() {
        int index = modelComboBox.getSelectedIndex();
        // Mapeia o índice para o nome da carta: 0->"SVM", 1->"J48", 2->"AdaBoost", 3->"RandomForest", 4->"IBk", 5->"RBF"
        switch (index) {
            case 0:
                tuningCardLayout.show(tuningPanel, "SVM");
                break;
            case 1:
                tuningCardLayout.show(tuningPanel, "J48");
                break;
            case 2:
                tuningCardLayout.show(tuningPanel, "AdaBoost");
                break;
            case 3:
                tuningCardLayout.show(tuningPanel, "RandomForest");
                break;
            case 4:
                tuningCardLayout.show(tuningPanel, "IBk");
                break;
            case 5:
                tuningCardLayout.show(tuningPanel, "RBF");
                break;
            default:
                break;
        }
    }

    private void trainAndEvaluate() {
        try {
            // Carrega a base de dados (ex.: Iris normalizada e balanceada)
            classifier.loadData("data/Iris.csv");

            // Determina a opção de modelo (1 a 6)
            int modelOption = modelComboBox.getSelectedIndex() + 1;
            double[] tuningParams;
            switch (modelOption) {
                case 1: // SVM
                    tuningParams = new double[2];
                    tuningParams[0] = Double.parseDouble(svmCField.getText());
                    tuningParams[1] = Double.parseDouble(svmKernelExpField.getText());
                    break;
                case 2: // J48
                    tuningParams = new double[2];
                    tuningParams[0] = Double.parseDouble(j48ConfidenceField.getText());
                    tuningParams[1] = Double.parseDouble(j48MinNumField.getText());
                    break;
                case 3: // AdaBoost
                    tuningParams = new double[1];
                    tuningParams[0] = Double.parseDouble(adaBoostIterationsField.getText());
                    break;
                case 4: // RandomForest
                    tuningParams = new double[1];
                    tuningParams[0] = Double.parseDouble(rfNumTreesField.getText());
                    break;
                case 5: // IBk
                    tuningParams = new double[1];
                    tuningParams[0] = Double.parseDouble(ibkKField.getText());
                    break;
                case 6: // RBF
                    tuningParams = new double[3];
                    tuningParams[0] = Double.parseDouble(rbfHiddenField.getText());
                    tuningParams[1] = Double.parseDouble(rbfSeedField.getText());
                    tuningParams[2] = Double.parseDouble(rbfUseCGDFied.getText());
                    break;
                default:
                    throw new IllegalArgumentException("Seleção de modelo inválida.");
            }
            // Configura o classificador com os parâmetros
            classifier.setClassifier(modelOption, tuningParams);

            // Executa avaliação conforme método selecionado
            if (trainTestRadio.isSelected()) {
                double trainPercentage = Double.parseDouble(trainPercentageField.getText());
                classifier.trainTestSplit(trainPercentage);
                classifier.trainClassifier();
                classifier.evaluateModel();
            } else {
                // Para CV, treina com os dados completos e avalia com o número de folds informado
                classifier.trainClassifier();
                int folds = Integer.parseInt(foldsField.getText());
                classifier.evaluateModelCV(folds);
            }

            // Exibe resultados da avaliação
            Evaluation eval = classifier.getEvaluation();
            String evalText = eval.toSummaryString("\nResultados\n======\n", true)
                    + "\n" + eval.toClassDetailsString()
                    + "\n" + eval.toMatrixString();
            evaluationTextArea.setText(evalText);
        } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(this, "Erro: " + ex.getMessage(), "Erro", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void evaluateMultipleCVAction() {
        try {
            // Carrega a base de dados (normalizada e balanceada)
            classifier.loadData("data/Iris.csv");

            // Determina a opção de modelo e configura parâmetros (como no método anterior)
            int modelOption = modelComboBox.getSelectedIndex() + 1;
            double[] tuningParams;
            switch (modelOption) {
                case 1:
                    tuningParams = new double[2];
                    tuningParams[0] = Double.parseDouble(svmCField.getText());
                    tuningParams[1] = Double.parseDouble(svmKernelExpField.getText());
                    break;
                case 2:
                    tuningParams = new double[2];
                    tuningParams[0] = Double.parseDouble(j48ConfidenceField.getText());
                    tuningParams[1] = Double.parseDouble(j48MinNumField.getText());
                    break;
                case 3:
                    tuningParams = new double[1];
                    tuningParams[0] = Double.parseDouble(adaBoostIterationsField.getText());
                    break;
                case 4:
                    tuningParams = new double[1];
                    tuningParams[0] = Double.parseDouble(rfNumTreesField.getText());
                    break;
                case 5:
                    tuningParams = new double[1];
                    tuningParams[0] = Double.parseDouble(ibkKField.getText());
                    break;
                case 6:
                    tuningParams = new double[1];
                    tuningParams[0] = Double.parseDouble(rbfHiddenField.getText());
                    break;
                default:
                    throw new IllegalArgumentException("Seleção de modelo inválida.");
            }
            // Configura o classificador
            classifier.setClassifier(modelOption, tuningParams);
            // Para avaliação múltipla, usamos validação cruzada para os k: 5, 10, 15, 20, 25
            int[] foldsArray = {5, 10, 15, 20, 25};
            String table = classifier.evaluateMultipleCV(foldsArray);
            evaluationTextArea.setText(table);
        } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(this, "Erro: " + ex.getMessage(), "Erro", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void showModelSummary() {
        try {
            String summary = (classifier.getEvaluation() != null)
                    ? classifier.getEvaluation().toSummaryString()
                    : "Modelo não treinado ainda.";
            JOptionPane.showMessageDialog(this, summary, "Resumo do Modelo", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
