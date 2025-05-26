import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

export interface PPOConfig {
  learningRate: number;
  gamma: number; // discount factor
  lambdaGAE: number; // GAE parameter
  clipEpsilon: number; // PPO clipping parameter
  entropyCoef: number; // entropy coefficient
  valueCoef: number; // value function coefficient
  batchSize: number;
  epochs: number; // training epochs per update
}

export interface SudokuState {
  board: number[][];
  validMoves: { row: number; col: number; value: number; }[];
}

export interface TrainingStep {
  state: number[];
  action: number;
  reward: number;
  value: number;
  logProb: number;
  done: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class PPOAgentService {
  private actor: tf.LayersModel | null = null;
  private critic: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null; // Single optimizer for combined training
  private config: PPOConfig;
  private trainingBuffer: TrainingStep[] = [];
  private totalSteps = 0;
  private isTraining = false;

  constructor() {
    this.config = {
      learningRate: 0.0003,
      gamma: 0.99,
      lambdaGAE: 0.95,
      clipEpsilon: 0.2,
      entropyCoef: 0.01,
      valueCoef: 0.5,
      batchSize: 64,
      epochs: 4
    };
  }

  async initialize(config?: Partial<PPOConfig>): Promise<void> {
    if (config) {
      this.config = { ...this.config, ...config };
    }

    // Initialize single optimizer for combined training
    this.optimizer = tf.train.adam(this.config.learningRate);

    // Initialize actor network (policy)
    this.actor = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [81], units: 256, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 729, activation: 'softmax' }) // 81 cells * 9 possible values
      ]
    });

    // Initialize critic network (value function)
    this.critic = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [81], units: 256, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 1, activation: 'linear' })
      ]
    });

    // Note: We don't compile the models since we'll use custom training loops
    console.log('PPO Agent initialized successfully');
  }

  // Convert Sudoku board to state vector
  private boardToState(board: number[][]): number[] {
    const state: number[] = [];
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        state.push(board[i][j] / 9.0); // Normalize to [0, 1]
      }
    }
    return state;
  }

  // Get valid actions for current state
  private getValidActions(board: number[][]): number[] {
    const validActions: number[] = [];
    
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (board[row][col] === 0) { // Empty cell
          for (let value = 1; value <= 9; value++) {
            if (this.isValidMove(board, row, col, value)) {
              const action = row * 81 + col * 9 + (value - 1);
              validActions.push(action);
            }
          }
        }
      }
    }
    
    return validActions;
  }

  // Check if a move is valid
  private isValidMove(board: number[][], row: number, col: number, value: number): boolean {
    // Check row
    for (let c = 0; c < 9; c++) {
      if (board[row][c] === value) return false;
    }

    // Check column
    for (let r = 0; r < 9; r++) {
      if (board[r][col] === value) return false;
    }

    // Check 3x3 box
    const boxRow = Math.floor(row / 3) * 3;
    const boxCol = Math.floor(col / 3) * 3;
    for (let r = boxRow; r < boxRow + 3; r++) {
      for (let c = boxCol; c < boxCol + 3; c++) {
        if (board[r][c] === value) return false;
      }
    }

    return true;
  }

  // Enhanced action selection with exploration bonus for early training
  async selectAction(board: number[][]): Promise<{ action: number; logProb: number; value: number }> {
    if (!this.actor || !this.critic) {
      throw new Error('PPO Agent not initialized');
    }

    const state = this.boardToState(board);
    const validActions = this.getValidActions(board);
    
    if (validActions.length === 0) {
      throw new Error('No valid actions available');
    }

    // Get policy distribution
    const stateTensor = tf.tensor2d([state]);
    const policyOutput = this.actor.predict(stateTensor) as tf.Tensor;
    const valueOutput = this.critic.predict(stateTensor) as tf.Tensor;

    const policyData = await policyOutput.data();
    const valueData = await valueOutput.data();

    // Create masked policy (only valid actions)
    const maskedPolicy = new Float32Array(729);
    let sum = 0;
    
    for (const action of validActions) {
      maskedPolicy[action] = policyData[action];
      sum += policyData[action];
    }

    // Enhanced normalization with exploration bonus for early training
    if (sum > 0) {
      for (const action of validActions) {
        maskedPolicy[action] /= sum;
      }
    } else {
      // Uniform distribution if all probabilities are 0
      const uniform = 1.0 / validActions.length;
      for (const action of validActions) {
        maskedPolicy[action] = uniform;
      }
    }

    // Add exploration bonus for cells with fewer constraints (easier to fill)
    const explorationBonuses = this.calculateExplorationBonuses(board, validActions);
    for (let i = 0; i < validActions.length; i++) {
      const action = validActions[i];
      maskedPolicy[action] = maskedPolicy[action] * 0.7 + explorationBonuses[i] * 0.3;
    }

    // Renormalize after adding exploration bonuses
    sum = 0;
    for (const action of validActions) {
      sum += maskedPolicy[action];
    }
    if (sum > 0) {
      for (const action of validActions) {
        maskedPolicy[action] /= sum;
      }
    }

    // Sample action using accumulated probabilities
    let random = Math.random();
    let selectedAction = validActions[0];
    let logProb = Math.log(maskedPolicy[selectedAction] + 1e-8);

    let accumulated = 0;
    for (const action of validActions) {
      accumulated += maskedPolicy[action];
      if (random <= accumulated) {
        selectedAction = action;
        logProb = Math.log(maskedPolicy[action] + 1e-8);
        break;
      }
    }

    // Cleanup tensors
    stateTensor.dispose();
    policyOutput.dispose();
    valueOutput.dispose();

    return {
      action: selectedAction,
      logProb: logProb,
      value: valueData[0]
    };
  }

  // Calculate exploration bonuses to encourage filling cells with fewer constraints
  private calculateExplorationBonuses(board: number[][], validActions: number[]): number[] {
    const bonuses: number[] = [];
    
    for (const action of validActions) {
      const row = Math.floor(action / 81);
      const col = Math.floor((action % 81) / 9);
      const value = (action % 9) + 1;
      
      // Count how many other cells this move would affect (constraint cells)
      let constraintCells = 0;
      
      // Count empty cells in same row
      for (let c = 0; c < 9; c++) {
        if (c !== col && board[row][c] === 0) constraintCells++;
      }
      
      // Count empty cells in same column  
      for (let r = 0; r < 9; r++) {
        if (r !== row && board[r][col] === 0) constraintCells++;
      }
      
      // Count empty cells in same 3x3 box
      const boxRow = Math.floor(row / 3) * 3;
      const boxCol = Math.floor(col / 3) * 3;
      for (let r = boxRow; r < boxRow + 3; r++) {
        for (let c = boxCol; c < boxCol + 3; c++) {
          if ((r !== row || c !== col) && board[r][c] === 0) constraintCells++;
        }
      }
      
      // Higher bonus for moves that affect more cells (more strategic)
      const bonus = Math.min(1.0, (constraintCells + 1) / 20.0);
      bonuses.push(bonus);
    }
    
    return bonuses;
  }

  // Apply action to board with enhanced reward system
  applyAction(board: number[][], action: number): { newBoard: number[][]; reward: number; done: boolean } {
    const row = Math.floor(action / 81);
    const col = Math.floor((action % 81) / 9);
    const value = (action % 9) + 1;

    const newBoard = board.map(row => [...row]);
    newBoard[row][col] = value;

    // Enhanced reward calculation
    let reward = 0;
    
    // Base reward for valid placement
    reward += 2.0;
    
    // Strategic placement bonuses
    const beforeValidMoves = this.getValidActions(board).length;
    const afterValidMoves = this.getValidActions(newBoard).length;
    
    // Bonus for maintaining or creating more possibilities
    if (afterValidMoves > beforeValidMoves) {
      reward += 1.0; // Opened up new possibilities
    } else if (afterValidMoves === beforeValidMoves) {
      reward += 0.5; // Maintained same level of possibilities
    }
    
    // Bonus for solving cells that eliminate possibilities strategically
    const eliminatedPossibilities = this.countEliminatedPossibilities(board, newBoard, row, col, value);
    reward += eliminatedPossibilities * 0.2;
    
    // Bonus for filling cells with fewer initial candidates (harder cells)
    const initialCandidates = this.countInitialCandidates(board, row, col);
    if (initialCandidates <= 2) {
      reward += 3.0; // High bonus for solving difficult cells
    } else if (initialCandidates <= 4) {
      reward += 1.5; // Medium bonus
    } else {
      reward += 0.5; // Small bonus for easy cells
    }
    
    // Bonus for completing regions (rows, columns, boxes)
    reward += this.calculateCompletionBonuses(newBoard, row, col);
    
    // Check if puzzle is solved
    const done = this.isPuzzleSolved(newBoard);
    if (done) {
      reward += 50.0; // Large bonus for solving the entire puzzle
    }

    // Small step penalty to encourage efficiency (reduced from before)
    reward -= 0.005;

    return { newBoard, reward, done };
  }

  // Count initial candidates for a cell
  private countInitialCandidates(board: number[][], row: number, col: number): number {
    let candidates = 0;
    for (let value = 1; value <= 9; value++) {
      if (this.isValidMove(board, row, col, value)) {
        candidates++;
      }
    }
    return candidates;
  }

  // Calculate bonuses for completing rows, columns, or boxes
  private calculateCompletionBonuses(board: number[][], row: number, col: number): number {
    let bonus = 0;
    
    // Check if row is complete
    let rowComplete = true;
    for (let c = 0; c < 9; c++) {
      if (board[row][c] === 0) {
        rowComplete = false;
        break;
      }
    }
    if (rowComplete) bonus += 5.0;
    
    // Check if column is complete
    let colComplete = true;
    for (let r = 0; r < 9; r++) {
      if (board[r][col] === 0) {
        colComplete = false;
        break;
      }
    }
    if (colComplete) bonus += 5.0;
    
    // Check if 3x3 box is complete
    const boxRow = Math.floor(row / 3) * 3;
    const boxCol = Math.floor(col / 3) * 3;
    let boxComplete = true;
    for (let r = boxRow; r < boxRow + 3; r++) {
      for (let c = boxCol; c < boxCol + 3; c++) {
        if (board[r][c] === 0) {
          boxComplete = false;
          break;
        }
      }
      if (!boxComplete) break;
    }
    if (boxComplete) bonus += 5.0;
    
    return bonus;
  }

  // Count how many possibilities are eliminated by placing a value
  private countEliminatedPossibilities(oldBoard: number[][], newBoard: number[][], row: number, col: number, value: number): number {
    let eliminated = 0;
    
    // Count eliminations in row
    for (let c = 0; c < 9; c++) {
      if (c !== col && oldBoard[row][c] === 0) {
        eliminated++;
      }
    }
    
    // Count eliminations in column
    for (let r = 0; r < 9; r++) {
      if (r !== row && oldBoard[r][col] === 0) {
        eliminated++;
      }
    }
    
    // Count eliminations in 3x3 box
    const boxRow = Math.floor(row / 3) * 3;
    const boxCol = Math.floor(col / 3) * 3;
    
    for (let r = boxRow; r < boxRow + 3; r++) {
      for (let c = boxCol; c < boxCol + 3; c++) {
        if ((r !== row || c !== col) && oldBoard[r][c] === 0) {
          eliminated++;
        }
      }
    }
    
    return eliminated;
  }

  // Check if puzzle is solved
  private isPuzzleSolved(board: number[][]): boolean {
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        if (board[i][j] === 0) {
          return false;
        }
      }
    }
    return true;
  }

  // Store training step
  storeStep(step: TrainingStep): void {
    this.trainingBuffer.push(step);
    this.totalSteps++;
  }

  // Calculate advantages using GAE
  private calculateAdvantages(steps: TrainingStep[]): { advantages: number[]; returns: number[] } {
    const advantages: number[] = [];
    const returns: number[] = [];
    
    let nextValue = 0;
    let advantage = 0;
    
    // Calculate advantages in reverse order
    for (let i = steps.length - 1; i >= 0; i--) {
      const step = steps[i];
      const delta = step.reward + this.config.gamma * nextValue * (1 - (step.done ? 1 : 0)) - step.value;
      advantage = delta + this.config.gamma * this.config.lambdaGAE * advantage * (1 - (step.done ? 1 : 0));
      advantages.unshift(advantage);
      
      const ret = advantage + step.value;
      returns.unshift(ret);
      
      nextValue = step.value;
    }
    
    return { advantages, returns };
  }

  // Train the agent with combined PPO loss
  async train(): Promise<{ actorLoss: number; criticLoss: number; avgAdvantage: number }> {
    if (!this.actor || !this.critic || !this.optimizer) {
      throw new Error('PPO Agent not initialized');
    }

    if (this.trainingBuffer.length < this.config.batchSize) {
      return { actorLoss: 0, criticLoss: 0, avgAdvantage: 0 };
    }

    this.isTraining = true;

    const steps = [...this.trainingBuffer];
    this.trainingBuffer = [];

    const { advantages, returns } = this.calculateAdvantages(steps);
    
    // Normalize advantages
    const advMean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const advStd = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - advMean, 2), 0) / advantages.length);
    const normalizedAdvantages = advantages.map(adv => (adv - advMean) / (advStd + 1e-8));

    // Prepare training data
    const states = steps.map(step => step.state);
    const actions = steps.map(step => step.action);
    const oldLogProbs = steps.map(step => step.logProb);
    const oldValues = steps.map(step => step.value);

    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    let batchCount = 0;

    // Train for multiple epochs
    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      // Shuffle data
      const indices = Array.from({length: steps.length}, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Train in batches
      for (let i = 0; i < states.length; i += this.config.batchSize) {
        const batchEnd = Math.min(i + this.config.batchSize, states.length);
        const batchIndices = indices.slice(i, batchEnd);
        
        const batchStates = batchIndices.map(idx => states[idx]);
        const batchActions = batchIndices.map(idx => actions[idx]);
        const batchAdvantages = batchIndices.map(idx => normalizedAdvantages[idx]);
        const batchReturns = batchIndices.map(idx => returns[idx]);
        const batchOldLogProbs = batchIndices.map(idx => oldLogProbs[idx]);
        const batchOldValues = batchIndices.map(idx => oldValues[idx]);

        // Train with combined PPO loss
        const losses = await this.trainCombined(
          batchStates, 
          batchActions, 
          batchAdvantages, 
          batchReturns, 
          batchOldLogProbs,
          batchOldValues
        );
        
        totalPolicyLoss += losses.policyLoss;
        totalValueLoss += losses.valueLoss;
        batchCount++;
      }
    }

    this.isTraining = false;

    return {
      actorLoss: totalPolicyLoss / batchCount,
      criticLoss: totalValueLoss / batchCount,
      avgAdvantage: advMean
    };
  }

  // Combined PPO training with proper loss function
  private async trainCombined(
    states: number[][], 
    actions: number[], 
    advantages: number[], 
    returns: number[], 
    oldLogProbs: number[],
    oldValues: number[]
  ): Promise<{ policyLoss: number; valueLoss: number }> {
    if (!this.actor || !this.critic || !this.optimizer) {
      throw new Error('Models not initialized');
    }

    return tf.tidy(() => {
      // Convert to tensors with consistent float32 dtype
      const statesTensor = tf.tensor2d(states, undefined, 'float32');
      const actionIndices = tf.tensor1d(actions, 'int32');
      const advantagesTensor = tf.tensor1d(advantages, 'float32');
      const returnsTensor = tf.tensor1d(returns, 'float32');
      const oldLogProbsTensor = tf.tensor1d(oldLogProbs, 'float32');
      const oldValuesTensor = tf.tensor1d(oldValues, 'float32');

      // Combined loss function using tf.variableGrads
      const f = (): tf.Scalar => {
        // Forward pass through both networks
        const policyOutput = this.actor!.predict(statesTensor) as tf.Tensor;
        const valueOutput = this.critic!.predict(statesTensor) as tf.Tensor;
        const valuePreds = tf.squeeze(valueOutput);

        // === POLICY LOSS (PPO Clipped Surrogate) ===
        // Use one-hot encoding instead of gatherND for gradient support
        const actionOneHot = tf.oneHot(actionIndices, 729);
        const actionProbs = tf.sum(tf.mul(policyOutput, actionOneHot), 1);
        
        const actionProbsFloat32 = tf.cast(actionProbs, 'float32');
        const newLogProbs = tf.log(tf.add(actionProbsFloat32, tf.scalar(1e-8)));
        
        // Calculate ratio and clipped ratio
        const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbsTensor));
        const clippedRatio = tf.clipByValue(
          ratio, 
          1 - this.config.clipEpsilon, 
          1 + this.config.clipEpsilon
        );
        
        const surr1 = tf.mul(ratio, advantagesTensor);
        const surr2 = tf.mul(clippedRatio, advantagesTensor);
        const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

        // === VALUE LOSS (PPO Clipped Value) ===
        // Clip the value function similar to policy clipping
        const valueClipped = tf.add(
          oldValuesTensor,
          tf.clipByValue(
            tf.sub(valuePreds, oldValuesTensor),
            -this.config.clipEpsilon,
            this.config.clipEpsilon
          )
        );
        
        const valueLossUnclipped = tf.square(tf.sub(valuePreds, returnsTensor));
        const valueLossClipped = tf.square(tf.sub(valueClipped, returnsTensor));
        const valueLoss = tf.mean(tf.maximum(valueLossUnclipped, valueLossClipped));

        // === ENTROPY BONUS ===
        const epsilon = tf.scalar(1e-8);
        const policyOutputFloat32 = tf.cast(policyOutput, 'float32');
        const logPolicy = tf.log(tf.add(policyOutputFloat32, epsilon));
        const entropy = tf.neg(tf.sum(tf.mul(policyOutputFloat32, logPolicy), 1));
        const entropyBonus = tf.mean(entropy);

        // === COMBINED PPO LOSS ===
        // totalLoss = policyLoss + valueCoef * valueLoss - entropyCoef * entropyBonus
        const weightedValueLoss = tf.mul(tf.scalar(this.config.valueCoef), valueLoss);
        const weightedEntropy = tf.mul(tf.scalar(this.config.entropyCoef), entropyBonus);
        const totalLoss = tf.add(tf.sub(policyLoss, weightedEntropy), weightedValueLoss);

        return totalLoss as tf.Scalar;
      };

      // Compute gradients and apply to both networks simultaneously
      const { value: totalLoss, grads } = tf.variableGrads(f);
      
      // Apply gradients to both actor and critic using single optimizer
      this.optimizer!.applyGradients(grads);

      // Calculate individual loss components for monitoring
      const { policyLoss, valueLoss } = this.calculateLossComponents(
        statesTensor, actionIndices, advantagesTensor, returnsTensor, 
        oldLogProbsTensor, oldValuesTensor
      );

      const totalLossValue = totalLoss.dataSync()[0];
      
      // Cleanup tensors
      statesTensor.dispose();
      actionIndices.dispose();
      advantagesTensor.dispose();
      returnsTensor.dispose();
      oldLogProbsTensor.dispose();
      oldValuesTensor.dispose();
      totalLoss.dispose();
      Object.values(grads).forEach(grad => grad.dispose());

      return { 
        policyLoss: policyLoss, 
        valueLoss: valueLoss 
      };
    });
  }

  // Helper to calculate individual loss components for monitoring
  private calculateLossComponents(
    statesTensor: tf.Tensor, 
    actionIndices: tf.Tensor, 
    advantagesTensor: tf.Tensor, 
    returnsTensor: tf.Tensor,
    oldLogProbsTensor: tf.Tensor,
    oldValuesTensor: tf.Tensor
  ): { policyLoss: number; valueLoss: number } {
    return tf.tidy(() => {
      const policyOutput = this.actor!.predict(statesTensor) as tf.Tensor;
      const valueOutput = this.critic!.predict(statesTensor) as tf.Tensor;
      const valuePreds = tf.squeeze(valueOutput);

      // Policy loss calculation - use one-hot encoding instead of gatherND
      const actionOneHot = tf.oneHot(actionIndices, 729);
      const actionProbs = tf.sum(tf.mul(policyOutput, actionOneHot), 1);
      const actionProbsFloat32 = tf.cast(actionProbs, 'float32');
      const newLogProbs = tf.log(tf.add(actionProbsFloat32, tf.scalar(1e-8)));
      
      const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbsTensor));
      const clippedRatio = tf.clipByValue(ratio, 1 - this.config.clipEpsilon, 1 + this.config.clipEpsilon);
      
      const surr1 = tf.mul(ratio, advantagesTensor);
      const surr2 = tf.mul(clippedRatio, advantagesTensor);
      const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

      // Value loss calculation with clipping
      const valueClipped = tf.add(
        oldValuesTensor,
        tf.clipByValue(
          tf.sub(valuePreds, oldValuesTensor),
          -this.config.clipEpsilon,
          this.config.clipEpsilon
        )
      );
      
      const valueLossUnclipped = tf.square(tf.sub(valuePreds, returnsTensor));
      const valueLossClipped = tf.square(tf.sub(valueClipped, returnsTensor));
      const valueLoss = tf.mean(tf.maximum(valueLossUnclipped, valueLossClipped));

      return {
        policyLoss: policyLoss.dataSync()[0],
        valueLoss: valueLoss.dataSync()[0]
      };
    });
  }

  // Get training statistics
  getTrainingStats(): { totalSteps: number; bufferSize: number; isTraining: boolean } {
    return {
      totalSteps: this.totalSteps,
      bufferSize: this.trainingBuffer.length,
      isTraining: this.isTraining
    };
  }

  // Save model
  async saveModel(path: string): Promise<void> {
    if (!this.actor || !this.critic) {
      throw new Error('Models not initialized');
    }

    await this.actor.save(`${path}/actor`);
    await this.critic.save(`${path}/critic`);
  }

  // Load model
  async loadModel(path: string): Promise<void> {
    this.actor = await tf.loadLayersModel(`${path}/actor`);
    this.critic = await tf.loadLayersModel(`${path}/critic`);
  }

  // Export model weights
  async exportWeights(): Promise<{ actor: any; critic: any }> {
    if (!this.actor || !this.critic) {
      throw new Error('Models not initialized');
    }

    const actorWeights = this.actor.getWeights().map(w => w.arraySync());
    const criticWeights = this.critic.getWeights().map(w => w.arraySync());

    return {
      actor: actorWeights,
      critic: criticWeights
    };
  }

  // Import model weights
  async importWeights(weights: { actor: any; critic: any }): Promise<void> {
    if (!this.actor || !this.critic) {
      throw new Error('Models not initialized');
    }

    const actorTensors = weights.actor.map((w: any) => tf.tensor(w));
    const criticTensors = weights.critic.map((w: any) => tf.tensor(w));

    this.actor.setWeights(actorTensors);
    this.critic.setWeights(criticTensors);

    // Dispose tensors
    actorTensors.forEach((t: tf.Tensor) => t.dispose());
    criticTensors.forEach((t: tf.Tensor) => t.dispose());
  }

  // Update configuration and reinitialize optimizer if learning rate changed
  updateConfig(newConfig: Partial<PPOConfig>): void {
    const oldLearningRate = this.config.learningRate;
    this.config = { ...this.config, ...newConfig };
    
    // Reinitialize optimizer if learning rate changed
    if (newConfig.learningRate && newConfig.learningRate !== oldLearningRate) {
      if (this.optimizer) {
        this.optimizer.dispose();
      }
      this.optimizer = tf.train.adam(this.config.learningRate);
    }
  }

  // Get current configuration
  getConfig(): PPOConfig {
    return { ...this.config };
  }

  // Dispose of models and optimizer
  dispose(): void {
    if (this.actor) {
      this.actor.dispose();
      this.actor = null;
    }
    if (this.critic) {
      this.critic.dispose();
      this.critic = null;
    }
    if (this.optimizer) {
      this.optimizer.dispose();
      this.optimizer = null;
    }
  }

  // Validate and convert input to number
  private validateNumber(value: any, defaultValue: number): number {
    const num = Number(value);
    return isNaN(num) ? defaultValue : num;
  }
}