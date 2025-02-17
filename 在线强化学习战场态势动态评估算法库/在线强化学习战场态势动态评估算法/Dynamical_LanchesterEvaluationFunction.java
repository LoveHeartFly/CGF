package ai.evaluation;

import rts.GameState;
import rts.PhysicalGameState;
import rts.units.*;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

public class Dynamical_LanchesterEvaluationFunction extends EvaluationFunction {    
    public float[] W_BASE;
    public float[] W_RAX;  
    public float[] W_WORKER;
    public float[] W_LIGHT;
    public float[] W_RANGE;
    public float[] W_HEAVY;
    public float[] W_MINERALS_CARRIED;
    public float[] W_MINERALS_MINED;
    
    public float order = 1.7f;
    
    // AdamW参数
    private float beta1 = 0.9f; // 动量项系数
    private float beta2 = 0.999f; // 二阶动量项系数
    private float epsilon = 1e-8f; // 防止分母为零
    private float[] m_lr = new float[]{1e-4f}; // 学习率的一阶动量项
    private float[] v_lr = new float[]{1e-4f}; // 学习率的二阶动量项
    private float[] m_dr = new float[]{0.01f}; // 衰减率的一阶动量项
    private float[] v_dr = new float[]{0.01f}; // 衰减率的二阶动量项
    private int step = 0; // 记录当前步数
    
    // 记录上一次的各兵种得分
    private float lastBaseScore=1f;
    private float lastBarracksScore=1f;
    private float lastWorkerScore=1f;
    private float lastLightScore=1f;
    private float lastRangedScore=1f;  
    private float lastHeavyScore=1f;
    
    public Dynamical_LanchesterEvaluationFunction() {
        // 初始化权重数组
        W_BASE = new float[]{0.12900641042498262f, 0.48944975377829392f};
        W_RAX = new float[]{0.23108197488337265f, 0.55022866772062451f};
        W_WORKER = new float[]{0.18122298329807154f, -0.0078514695699861588f};
        W_LIGHT = new float[]{1.7496678034331925f, 0.12587241165484406f};
        W_RANGE = new float[]{1.6793840344563218f, 0.029918374064639004f};
        W_HEAVY = new float[]{3.9012441116439427f, 0.16414240458460899f};
        W_MINERALS_CARRIED = new float[]{0.3566229669443759f, 0.01061490087512941f};
        W_MINERALS_MINED = new float[]{0.30141654836442761f, 0.38643842595899713f};
    }
    
    public static float sigmoid(float x) {
        return (float) (1.0f / (1.0f + Math.pow(Math.E, -x)));
    }
    
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        long startTime = System.nanoTime();
        
        float score_max = base_score(maxplayer, gs);
        float score_min = base_score(minplayer, gs);
        float score = 2.0f * sigmoid(score_max - score_min) - 1.0f;
        
        logScore("Dynamical_LanchesterEvaluationFunction_base_score_max", score_max);
        logScore("Dynamical_LanchesterEvaluationFunction_base_score_min", score_min);
        logScore("Dynamical_LanchesterEvaluationFunction_eval", score);
        
        long endTime = System.nanoTime();
        long duration = endTime - startTime;
        
        logEvaluationTime(duration);
        
        return score;
    }

    public float base_score(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int index = 0;
        if (pgs.getWidth() == 128) {
            index = 1;
        }

        float score = 0.0f;
        float score_buildings = 0.0f;
        float nr_units = 0.0f;
        float res_carried = 0.0f;
        
        float baseScore = 0.0f;
        float barracksScore = 0.0f;
        float workerScore = 0.0f;
        float lightScore = 0.0f;
        float rangedScore = 0.0f;
        float heavyScore = 0.0f;
        
        UnitTypeTable utt = gs.getUnitTypeTable();
        
        for(Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                res_carried += u.getResources();
                switch (u.getType().name) {
                    case "Base":
                        baseScore = W_BASE[index] * u.getHitPoints();
                        score_buildings += baseScore;
                        break;
                    case "Barracks":
                        barracksScore = W_RAX[index] * u.getHitPoints();
                        score_buildings += barracksScore;
                        break;
                    case "Worker":
                        nr_units += 1;
                        workerScore = W_WORKER[index] * u.getHitPoints();
                        score += workerScore;
                        break;
                    case "Light":
                        nr_units += 1;
                        lightScore = W_LIGHT[index] * u.getHitPoints() / (float) u.getMaxHitPoints();
                        score += lightScore;
                        break;
                    case "Ranged":
                        nr_units += 1;
                        rangedScore = W_RANGE[index] * u.getHitPoints();
                        score += rangedScore;
                        break;
                    case "Heavy":
                        nr_units += 1;
                        heavyScore = W_HEAVY[index] * u.getHitPoints() / (float) u.getMaxHitPoints();
                        score += heavyScore;
                        break;
                }
            }
        }
                
        // 动态调整权重
        updateWeights(baseScore, barracksScore, workerScore, lightScore, rangedScore, heavyScore, score);
        
        score = (float) (score * Math.pow(nr_units, order - 1));
        score += score_buildings + res_carried * W_MINERALS_CARRIED[index] + gs.getPlayer(player).getResources() * W_MINERALS_MINED[index];

        return score;
    }

    private float calculateDelta(float currentScore, float lastScore) {
        if (lastScore == 0) {
            return (currentScore == 0) ? 0 : (currentScore > 0 ? 1 : -1);
        }
        return (currentScore - lastScore) / lastScore;
    }

    private void updateWeights(float baseScore, float barracksScore, float workerScore, 
                            float lightScore, float rangedScore, float heavyScore, float currentScore) {


        // Calculate score changes
        float baseDelta = calculateDelta(baseScore, lastBaseScore);
        float barracksDelta = calculateDelta(barracksScore, lastBarracksScore);
        float workerDelta = calculateDelta(workerScore, lastWorkerScore);
        float lightDelta = calculateDelta(lightScore, lastLightScore);
        float rangedDelta = calculateDelta(rangedScore, lastRangedScore);
        float heavyDelta = calculateDelta(heavyScore, lastHeavyScore);
        
        float scoreDelta = calculateDelta(currentScore, lastBaseScore + lastBarracksScore + lastWorkerScore + lastLightScore + lastRangedScore + lastHeavyScore);

        // Update weights using online reinforcement learning
        float lr = getAdamWLearningRate(scoreDelta);
        float dr = getAdamWDecayRate(scoreDelta);
        
        W_BASE[0] += lr * baseDelta;
        W_BASE[1] += lr * baseDelta;
        W_BASE[0] *= (1 - dr);
        W_BASE[1] *= (1 - dr);
        
        W_RAX[0] += lr * barracksDelta;
        W_RAX[1] += lr * barracksDelta;
        W_RAX[0] *= (1 - dr);
        W_RAX[1] *= (1 - dr);
        
        W_WORKER[0] += lr * workerDelta;
        W_WORKER[1] += lr * workerDelta;
        W_WORKER[0] *= (1 - dr);
        W_WORKER[1] *= (1 - dr);

        W_LIGHT[0] += lr * lightDelta;
        W_LIGHT[1] += lr * lightDelta;
        W_LIGHT[0] *= (1 - dr);
        W_LIGHT[1] *= (1 - dr);
        
        W_RANGE[0] += lr * rangedDelta;
        W_RANGE[1] += lr * rangedDelta;
        W_RANGE[0] *= (1 - dr);
        W_RANGE[1] *= (1 - dr);
        
        W_HEAVY[0] += lr * heavyDelta;
        W_HEAVY[1] += lr * heavyDelta;
        W_HEAVY[0] *= (1 - dr);
        W_HEAVY[1] *= (1 - dr);
        
        // Update last scores
        lastBaseScore = baseScore;
        lastBarracksScore = barracksScore;
        lastWorkerScore = workerScore;
        lastLightScore = lightScore;
        lastRangedScore = rangedScore;
        lastHeavyScore = heavyScore;
    }

    private float getAdamWLearningRate(float scoreDelta) {
        step++;
        m_lr[0] = beta1 * m_lr[0] + (1 - beta1) * scoreDelta;
        v_lr[0] = beta2 * v_lr[0] + (1 - beta2) * scoreDelta * scoreDelta;
        float m_hat = m_lr[0] / (1 - (float)Math.pow(beta1, step));
        float v_hat = v_lr[0] / (1 - (float)Math.pow(beta2, step));
        return m_hat / (float)(Math.sqrt(v_hat) + epsilon);
    }

    private float getAdamWDecayRate(float scoreDelta) {
        step++;
        m_dr[0] = beta1 * m_dr[0] + (1 - beta1) * scoreDelta;
        v_dr[0] = beta2 * v_dr[0] + (1 - beta2) * scoreDelta * scoreDelta;
        float m_hat = m_dr[0] / (1 - (float)Math.pow(beta1, step));
        float v_hat = v_dr[0] / (1 - (float)Math.pow(beta2, step));
        return m_hat / (float)(Math.sqrt(v_hat) + epsilon);
    }

    public float upperBound(GameState gs) {
        return 2.0f;
    }

    private void logScore(String filename, float score) {
        try (FileWriter writer = new FileWriter("./results/" + filename + ".txt", true)) {
            writer.write(score + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void logEvaluationTime(long duration) {
        try {
            File file = new File("./results/Dynamical_LanchesterEvaluationFunction.txt");
            boolean fileIsEmpty = file.length() == 0;

            try (FileWriter writer = new FileWriter(file, true)) {
                if (fileIsEmpty) {
                    writer.write("Evaluation time in nanoseconds:\n");
                }
                writer.write(duration + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
