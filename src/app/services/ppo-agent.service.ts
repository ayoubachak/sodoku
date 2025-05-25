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
  private actorOptimizer: tf.Optimizer | null = null;
  private criticOptimizer: tf.Optimizer | null = null;
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

    // Initialize optimizers
    this.actorOptimizer = tf.train.adam(this.config.learningRate);
    this.criticOptimizer = tf.train.adam(this.config.learningRate);

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

  // Select action using policy
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

    // Normalize
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

    // Sample action
    let random = Math.random();
    let selectedAction = validActions[0];
    let logProb = Math.log(maskedPolicy[selectedAction] + 1e-8);

    for (const action of validActions) {
      if (random <= maskedPolicy[action]) {
        selectedAction = action;
        logProb = Math.log(maskedPolicy[action] + 1e-8);
        break;
      }
      random -= maskedPolicy[action];
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

  // Apply action to board
  applyAction(board: number[][], action: number): { newBoard: number[][]; reward: number; done: boolean } {
    const row = Math.floor(action / 81);
    const col = Math.floor((action % 81) / 9);
    const value = (action % 9) + 1;

    const newBoard = board.map(row => [...row]);
    newBoard[row][col] = value;

    // Calculate reward
    let reward = 0;
    
    // Positive reward for valid placement
    reward += 1.0;
    
    // Bonus for solving cells that eliminate possibilities for other cells
    const eliminatedPossibilities = this.countEliminatedPossibilities(board, newBoard, row, col, value);
    reward += eliminatedPossibilities * 0.1;
    
    // Check if puzzle is solved
    const done = this.isPuzzleSolved(newBoard);
    if (done) {
      reward += 10.0; // Large bonus for solving
    }

    // Small penalty for each step to encourage efficiency
    reward -= 0.01;

    return { newBoard, reward, done };
  }

  // Count how many possibilities are eliminated by placing a value
  private countEliminatedPossibilities(oldBoard: number[][], newBoard: number[][], row: number, col: number, value: number): number {
    let count = 0;
    
    // Check row
    for (let c = 0; c < 9; c++) {
      if (c !== col && oldBoard[row][c] === 0) count++;
    }
    
    // Check column
    for (let r = 0; r < 9; r++) {
      if (r !== row && oldBoard[r][col] === 0) count++;
    }
    
    // Check 3x3 box
    const boxRow = Math.floor(row / 3) * 3;
    const boxCol = Math.floor(col / 3) * 3;
    for (let r = boxRow; r < boxRow + 3; r++) {
      for (let c = boxCol; c < boxCol + 3; c++) {
        if ((r !== row || c !== col) && oldBoard[r][c] === 0) count++;
      }
    }
    
    return count;
  }

  // Check if puzzle is solved
  private isPuzzleSolved(board: number[][]): boolean {
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        if (board[i][j] === 0) return false;
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

  // Train the agent
  async train(): Promise<{ actorLoss: number; criticLoss: number; avgAdvantage: number }> {
    if (!this.actor || !this.critic) {
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

    let totalActorLoss = 0;
    let totalCriticLoss = 0;

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

        // Train critic
        const criticLoss = await this.trainCritic(batchStates, batchReturns);
        totalCriticLoss += criticLoss;

        // Train actor
        const actorLoss = await this.trainActor(batchStates, batchActions, batchAdvantages, batchOldLogProbs);
        totalActorLoss += actorLoss;
      }
    }

    this.isTraining = false;

    const numBatches = Math.ceil(states.length / this.config.batchSize) * this.config.epochs;
    return {
      actorLoss: totalActorLoss / numBatches,
      criticLoss: totalCriticLoss / numBatches,
      avgAdvantage: advMean
    };
  }

  // Train critic network
  private async trainCritic(states: number[][], returns: number[]): Promise<number> {
    if (!this.critic || !this.criticOptimizer) {
      throw new Error('Critic not initialized');
    }

    return tf.tidy(() => {
      // Ensure consistent float32 dtype
      const statesTensor = tf.tensor2d(states, undefined, 'float32');
      const returnsTensor = tf.tensor2d(returns, [returns.length, 1], 'float32');

      // Use tf.variableGrads to compute gradients and apply them
      const f = (): tf.Scalar => {
        const predictions = this.critic!.predict(statesTensor) as tf.Tensor;
        const loss = tf.losses.meanSquaredError(returnsTensor, predictions);
        // Return scalar without unnecessary casting
        return tf.mean(loss);
      };

      const { value: loss, grads } = tf.variableGrads(f);
      
      // Apply gradients to critic
      this.criticOptimizer!.applyGradients(grads);
      
      const lossValue = loss.dataSync()[0];
      
      // Cleanup
      statesTensor.dispose();
      returnsTensor.dispose();
      loss.dispose();
      Object.values(grads).forEach(grad => grad.dispose());

      return lossValue;
    });
  }

  // Train actor network with proper gradient application
  private async trainActor(states: number[][], actions: number[], advantages: number[], oldLogProbs: number[]): Promise<number> {
    if (!this.actor || !this.actorOptimizer) {
      throw new Error('Actor not initialized');
    }

    return tf.tidy(() => {
      // Ensure all tensors use float32 dtype
      const statesTensor = tf.tensor2d(states, undefined, 'float32');
      const actionIndices = tf.tensor1d(actions, 'int32');
      const advantagesTensor = tf.tensor1d(advantages, 'float32');
      const oldLogProbsTensor = tf.tensor1d(oldLogProbs, 'float32');

      // Use tf.variableGrads to compute gradients and apply them
      const f = (): tf.Scalar => {
        // Forward pass to get current policy
        const policyOutput = this.actor!.predict(statesTensor) as tf.Tensor;
        
        // Calculate new log probabilities for taken actions
        const batchSize = statesTensor.shape[0];
        const batchIndices = tf.range(0, batchSize, 1, 'int32');
        
        // Create indices tensor with correct dtype for stack operation
        const indices = tf.stack([batchIndices, actionIndices], 1);
        const actionProbs = tf.gatherND(policyOutput, indices);
        
        // Ensure actionProbs is float32 before log operation
        const actionProbsFloat32 = tf.cast(actionProbs, 'float32');
        const newLogProbs = tf.log(tf.add(actionProbsFloat32, tf.scalar(1e-8)));
        
        // Calculate ratio (new_prob / old_prob) - ensure both tensors are float32
        const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbsTensor));
        
        // Calculate PPO clipped surrogate loss - use direct numbers for clipping
        const oneMinusEpsilon = 1 - this.config.clipEpsilon;
        const onePlusEpsilon = 1 + this.config.clipEpsilon;
        const clippedRatio = tf.clipByValue(ratio, oneMinusEpsilon, onePlusEpsilon);
        
        const surr1 = tf.mul(ratio, advantagesTensor);
        const surr2 = tf.mul(clippedRatio, advantagesTensor);
        const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
        
        // Calculate entropy bonus - ensure float32 throughout
        const epsilon = tf.scalar(1e-8);
        const policyOutputFloat32 = tf.cast(policyOutput, 'float32');
        const logPolicy = tf.log(tf.add(policyOutputFloat32, epsilon));
        const entropy = tf.neg(tf.sum(tf.mul(policyOutputFloat32, logPolicy), 1));
        const entropyCoef = tf.scalar(this.config.entropyCoef);
        const entropyBonus = tf.mul(entropyCoef, tf.mean(entropy));
        
        // Total loss = policy loss - entropy bonus
        const totalLoss = tf.sub(policyLoss, entropyBonus);
        
        // Return scalar without unnecessary casting
        return tf.mean(totalLoss);
      };

      const { value: loss, grads } = tf.variableGrads(f);
      
      // Apply gradients to actor network
      this.actorOptimizer!.applyGradients(grads);
      
      const lossValue = loss.dataSync()[0];
      
      // Cleanup tensors
      statesTensor.dispose();
      actionIndices.dispose();
      advantagesTensor.dispose();
      oldLogProbsTensor.dispose();
      loss.dispose();
      Object.values(grads).forEach(grad => grad.dispose());

      return lossValue;
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

  // Update configuration and reinitialize optimizers if learning rate changed
  updateConfig(newConfig: Partial<PPOConfig>): void {
    const oldLearningRate = this.config.learningRate;
    this.config = { ...this.config, ...newConfig };
    
    // Reinitialize optimizers if learning rate changed
    if (newConfig.learningRate && newConfig.learningRate !== oldLearningRate) {
      if (this.actorOptimizer) {
        this.actorOptimizer.dispose();
      }
      if (this.criticOptimizer) {
        this.criticOptimizer.dispose();
      }
      this.actorOptimizer = tf.train.adam(this.config.learningRate);
      this.criticOptimizer = tf.train.adam(this.config.learningRate);
    }
  }

  // Get current configuration
  getConfig(): PPOConfig {
    return { ...this.config };
  }

  // Dispose of models and optimizers
  dispose(): void {
    if (this.actor) {
      this.actor.dispose();
      this.actor = null;
    }
    if (this.critic) {
      this.critic.dispose();
      this.critic = null;
    }
    if (this.actorOptimizer) {
      this.actorOptimizer.dispose();
      this.actorOptimizer = null;
    }
    if (this.criticOptimizer) {
      this.criticOptimizer.dispose();
      this.criticOptimizer = null;
    }
  }

  // Validate and convert input to number
  private validateNumber(value: any, defaultValue: number): number {
    const num = Number(value);
    return isNaN(num) ? defaultValue : num;
  }
}