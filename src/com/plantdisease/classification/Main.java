package com.plantdisease.classification;

import javax.swing.SwingUtilities;

public class Main {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            FrontUI ui = new FrontUI();
            ui.setVisible(true);
        });
    }
}

