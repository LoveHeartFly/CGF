package ai.evaluation;

import rts.GameState;
import rts.PhysicalGameState;
import rts.units.*;

import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * Dynamical_SimpleEvaluationFunction class to evaluate game states and log results to files.
 * Each method call logs the score in a file named after the method in the ./results/ directory.
 */
public class Dynamical_SimpleEvaluationFunction extends EvaluationFunction {
    public float RESOURCE;
    public float RESOURCE_IN_WORKER;
    public float UNIT_BONUS_MULTIPLIER;

    // 单位类型到权重的映射
    private Map<String, Float> unitWeights;

    // AdamW参数
    private float beta1 = 0.9f; // 动量项系数
    private float beta2 = 0.999f; // 二阶动量项系数
    private float epsilon = 1e-8f; // 防止分母为零
    private float[] m_lr = new float[]{1e-4f}; // 学习率的一阶动量项
    private float[] v_lr = new float[]{1e-4f}; // 学习率的二阶动量项
    private float[] m_dr = new float[]{0.01f}; // 衰减率的一阶动量项
    private float[] v_dr = new float[]{0.01f}; // 衰减率的二阶动量项
    private int step = 0; // 记录当前步数

    // 记录上一次的权重值
    private float lastResource;
    private float lastResourceInWorker;
    private float lastUnitBonusMultiplier;
    private float lastResourceScore;
    private float lastResourceInWorkerScore;
    private float lastUnitBonusMultiplierScore;
    private Map<String, Float> lastUnitWeights;

    public Dynamical_SimpleEvaluationFunction() {
        // 初始化权重值
        RESOURCE = 20f;
        RESOURCE_IN_WORKER = 10f;
        UNIT_BONUS_MULTIPLIER = 40f;

        // 初始化单位类型到权重的映射
        unitWeights = new HashMap<>();
        unitWeights.put("Base", 20f);
        unitWeights.put("Barracks", 20f);
        unitWeights.put("Worker", 20f);
        unitWeights.put("Light", 20f);
        unitWeights.put("Ranged", 20f);
        unitWeights.put("Heavy", 20f);

        // 记录初始权重值和分数
        lastResource = RESOURCE;
        lastResourceInWorker = RESOURCE_IN_WORKER;
        lastUnitBonusMultiplier = UNIT_BONUS_MULTIPLIER;
        lastResourceScore = 0f;
        lastResourceInWorkerScore = 0f;
        lastUnitBonusMultiplierScore = 0f;
        lastUnitWeights = new HashMap<>(unitWeights);
    }

    public static float sigmoid(float x) {
        return (float) (1.0f / (1.0f + Math.pow(Math.E, -x)));
    }

    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        long startTime = System.nanoTime();
        
        float score_max = base_score(maxplayer, gs);
        float score_min = base_score(minplayer, gs);
        float score = 2.0f * sigmoid(score_max - score_min) - 1.0f;
        
        logScore("Dynamical_SimpleSqrtEvaluationFunction_base_score_max", score_max);
        logScore("Dynamical_SimpleSqrtEvaluationFunction_base_score_min", score_min);
        logScore("Dynamical_SimpleSqrtEvaluationFunction_eval", score);
        
        long endTime = System.nanoTime();
        long duration = endTime - startTime;
        
        logEvaluationTime(duration);

        return score;
    }

    public float base_score(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        float score = gs.getPlayer(player).getResources() * RESOURCE;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                score += u.getResources() * RESOURCE_IN_WORKER;
                score += UNIT_BONUS_MULTIPLIER * (u.getCost() * u.getHitPoints()) / (float) u.getMaxHitPoints();
            }
        }

        // 动态调整权重
        updateWeights(score, pgs, player);

        return score;
    }

    public float upperBound(GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int free_resources = 0;
        int player_resources[] = { gs.getPlayer(0).getResources(), gs.getPlayer(1).getResources() };
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == -1)
                free_resources += u.getResources();
            if (u.getPlayer() == 0) {
                player_resources[0] += u.getResources();
                player_resources[0] += u.getCost();
            }
            if (u.getPlayer() == 1) {
                player_resources[1] += u.getResources();
                player_resources[1] += u.getCost();
            }
        }
        float score = (free_resources + Math.max(player_resources[0], player_resources[1])) * UNIT_BONUS_MULTIPLIER;
        return score;
    }

    private float calculateDelta(float currentScore, float lastScore) {
        if (lastScore == 0) {
            return (currentScore == 0) ? 0 : (currentScore > 0 ? 1 : -1);
        }
        return (currentScore - lastScore) / lastScore;
    }

    private void updateWeights(float currentScore, PhysicalGameState pgs, int player) {
        float scoreDelta = calculateDelta(currentScore,
                lastResourceScore + lastResourceInWorkerScore + lastUnitBonusMultiplierScore);

        // Update weights using online reinforcement learning
        float lr = getAdamWLearningRate(scoreDelta);
        float dr = getAdamWDecayRate(scoreDelta);

        RESOURCE += lr * scoreDelta;
        RESOURCE *= (1 - dr);

        RESOURCE_IN_WORKER += lr * scoreDelta;
        RESOURCE_IN_WORKER *= (1 - dr);

        UNIT_BONUS_MULTIPLIER += lr * scoreDelta;
        UNIT_BONUS_MULTIPLIER *= (1 - dr);

        // Update unit weights
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                String unitType = u.getType().name;
                float unitWeight = unitWeights.get(unitType);
                float lastUnitWeight = lastUnitWeights.get(unitType);
                float unitWeightDelta = calculateDelta(unitWeight, lastUnitWeight);

                unitWeights.put(unitType, unitWeight + lr * unitWeightDelta);
                unitWeights.put(unitType, unitWeights.get(unitType) * (1 - dr));
            }
        }

        // Record current scores and weights
        lastResourceScore = currentScore * RESOURCE;
        lastResourceInWorkerScore = currentScore * RESOURCE_IN_WORKER;
        lastUnitBonusMultiplierScore = currentScore * UNIT_BONUS_MULTIPLIER;
        lastUnitWeights = new HashMap<>(unitWeights);
    }

    private float getAdamWLearningRate(float scoreDelta) {
        step++;

        // Update biased first moment estimate
        m_lr[0] = beta1 * m_lr[0] + (1 - beta1) * scoreDelta;

        // Update biased second moment estimate
        v_lr[0] = beta2 * v_lr[0] + (1 - beta2) * scoreDelta * scoreDelta;

        // Bias correction
        float m_hat = m_lr[0] / (1 - (float) Math.pow(beta1, step));
        float v_hat = v_lr[0] / (1 - (float) Math.pow(beta2, step));

        // Update learning rate
        float lr = m_hat / (float) (Math.sqrt(v_hat) + epsilon);

        return lr;
    }

    private float getAdamWDecayRate(float scoreDelta) {
        step++;

        // Update biased first moment estimate
        m_dr[0] = beta1 * m_dr[0] + (1 - beta1) * scoreDelta;

        // Update biased second moment estimate
        v_dr[0] = beta2 * v_dr[0] + (1 - beta2) * scoreDelta * scoreDelta;

        // Bias correction
        float m_hat = m_dr[0] / (1 - (float) Math.pow(beta1, step));
        float v_hat = v_dr[0] / (1 - (float) Math.pow(beta2, step));

        // Update decay rate
        float dr = m_hat / (float) (Math.sqrt(v_hat) + epsilon);

        return dr;
    }

    // Helper method to log scores to a file
    private void logScore(String methodName, float score) {
        try (FileWriter fw = new FileWriter("./results/" + methodName + ".txt", true)) {
            fw.write("Score: " + score + "\n");
        } catch (IOException e) {
            System.err.println("Error writing score to file: " + e.getMessage());
        }
    }

    // Helper method to log evaluation time to a file
    private void logEvaluationTime(long duration) {
        try {
            File file = new File("./results/Dynamical_SimpleEvaluationFunction.txt");
            boolean isNewFile = file.createNewFile();
            try (FileWriter fw = new FileWriter(file, true)) {
                if (isNewFile || file.length() == 0) {
                    fw.write("Evaluation times (in nanoseconds):\n");
                }
                fw.write(duration + "\n");
            }
        } catch (IOException e) {
            System.err.println("Error writing evaluation time to file: " + e.getMessage());
        }
    }
}
