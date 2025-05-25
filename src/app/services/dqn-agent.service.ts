import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

export interface DQNConfig {
  learningRate: number;
  gamma: number; // discount factor
  epsilon: number; // exploration rate
  epsilonMin: number;
  epsilonDecay: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFreq: number; // how often to update target network
  doubleDQN: boolean;
  duelingDQN: boolean;
}

export interface DQNTrainingStep {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
  priority?: number; // for prioritized experience replay
}

@Injectable({
  providedIn: 'root'
})
export class DQNAgentService {
  private readonly replayBuffer: Experience[] = [];
  private mainNetwork!: tf.LayersModel;
  private targetNetwork!: tf.LayersModel;
  private updateCounter = 0; // Remove readonly since it needs to be incremented
  private totalSteps = 0; // Add missing property
  private isTraining = false; // Add missing property
  
  private config: DQNConfig = {
    learningRate: 0.0005,
    gamma: 0.99,
    epsilon: 1.0,
    epsilonMin: 0.01,
    epsilonDecay: 0.995,
    batchSize: 32,
    memorySize: 10000,
    targetUpdateFreq: 100,
    doubleDQN: true,
    duelingDQN: true
  };

  constructor() {}

  async initialize(config?: Partial<DQNConfig>): Promise<void> {
    if (config) {
      this.config = { ...this.config, ...config };
    }

    // Build the main network
    this.mainNetwork = this.buildNetwork();
    
    // Build the target network (copy of main network)
    this.targetNetwork = this.buildNetwork();
    
    // Copy weights from main to target network
    await this.updateTargetNetwork();

    console.log('DQN Agent initialized with dueling architecture');
  }

  private buildNetwork(): tf.LayersModel {
    if (this.config.duelingDQN) {
      return this.buildDuelingNetwork();
    } else {
      return this.buildStandardNetwork();
    }
  }

  private buildDuelingNetwork(): tf.LayersModel {
    // Input layer for board state (81 cells)
    const input = tf.input({ shape: [81] });
    
    // Shared feature layers
    const shared1 = tf.layers.dense({ units: 512, activation: 'relu' }).apply(input) as tf.SymbolicTensor;
    const dropout1 = tf.layers.dropout({ rate: 0.3 }).apply(shared1) as tf.SymbolicTensor;
    const shared2 = tf.layers.dense({ units: 256, activation: 'relu' }).apply(dropout1) as tf.SymbolicTensor;
    const dropout2 = tf.layers.dropout({ rate: 0.3 }).apply(shared2) as tf.SymbolicTensor;
    const shared3 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(dropout2) as tf.SymbolicTensor;

    // Value stream - estimates state value
    const valueStream = tf.layers.dense({ units: 64, activation: 'relu' }).apply(shared3) as tf.SymbolicTensor;
    const stateValue = tf.layers.dense({ units: 1, activation: 'linear', name: 'state_value' }).apply(valueStream) as tf.SymbolicTensor;

    // Advantage stream - estimates action advantages
    const advantageStream = tf.layers.dense({ units: 64, activation: 'relu' }).apply(shared3) as tf.SymbolicTensor;
    const actionAdvantages = tf.layers.dense({ units: 729, activation: 'linear', name: 'action_advantages' }).apply(advantageStream) as tf.SymbolicTensor;

    // Simplified dueling combination - just add value to advantages directly
    // This is a simplified version that avoids the lambda layer
    const qValues = tf.layers.add().apply([stateValue, actionAdvantages]) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: qValues });
    
    model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    return model;
  }

  private buildStandardNetwork(): tf.LayersModel {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [81], units: 512, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: 256, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 729, activation: 'linear' }) // 81 cells * 9 possible values
      ]
    });

    model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    return model;
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

  // Select action using epsilon-greedy policy
  async selectAction(board: number[][]): Promise<{ action: number; qValue: number; isExploration: boolean }> {
    if (!this.mainNetwork) {
      throw new Error('DQN Agent not initialized');
    }

    const state = this.boardToState(board);
    const validActions = this.getValidActions(board);
    
    if (validActions.length === 0) {
      throw new Error('No valid actions available');
    }

    let selectedAction: number;
    let qValue: number;
    let isExploration = false;

    // Epsilon-greedy action selection
    if (Math.random() < this.config.epsilon) {
      // Exploration: random valid action
      selectedAction = validActions[Math.floor(Math.random() * validActions.length)];
      isExploration = true;
      
      // Get Q-value for the random action
      const stateTensor = tf.tensor2d([state]);
      const qValues = this.mainNetwork.predict(stateTensor) as tf.Tensor;
      const qValuesData = await qValues.data();
      qValue = qValuesData[selectedAction];
      
      stateTensor.dispose();
      qValues.dispose();
    } else {
      // Exploitation: best valid action
      const stateTensor = tf.tensor2d([state]);
      const qValues = this.mainNetwork.predict(stateTensor) as tf.Tensor;
      const qValuesData = await qValues.data();

      // Find the best valid action
      let bestQValue = -Infinity;
      selectedAction = validActions[0];
      
      for (const action of validActions) {
        if (qValuesData[action] > bestQValue) {
          bestQValue = qValuesData[action];
          selectedAction = action;
        }
      }
      
      qValue = bestQValue;
      
      stateTensor.dispose();
      qValues.dispose();
    }

    return {
      action: selectedAction,
      qValue: qValue,
      isExploration: isExploration
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
    
    // Base reward for valid placement
    reward += 1.0;
    
    // Bonus for solving cells that eliminate possibilities for other cells
    const eliminatedPossibilities = this.countEliminatedPossibilities(board, newBoard, row, col, value);
    reward += eliminatedPossibilities * 0.1;
    
    // Check if puzzle is solved
    const done = this.isPuzzleSolved(newBoard);
    if (done) {
      reward += 50; // Large bonus for completing the puzzle
    }
    
    return { newBoard, reward, done };
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

  // Store experience in replay buffer
  storeExperience(experience: Experience): void {
    this.replayBuffer.push(experience);
    
    // Remove oldest experience if buffer is full
    if (this.replayBuffer.length > this.config.memorySize) {
      this.replayBuffer.shift();
    }
  }

  // Train the Q-network using experience replay
  async train(): Promise<{ qValue: number; epsilon: number }> {
    if (this.replayBuffer.length < this.config.batchSize) {
      return { qValue: 0, epsilon: this.config.epsilon };
    }

    try {
      this.isTraining = true;
      
      // Sample random batch from replay buffer
      const batch = this.sampleBatch(this.config.batchSize);
      
      // Prepare training data
      const states = batch.map(exp => exp.state);
      const nextStates = batch.map(exp => exp.nextState);
      
      const statesTensor = tf.tensor2d(states);
      const nextStatesTensor = tf.tensor2d(nextStates);
      
      // Get current Q-values and next Q-values
      const currentQValues = this.mainNetwork.predict(statesTensor) as tf.Tensor;
      const nextQValues = this.targetNetwork.predict(nextStatesTensor) as tf.Tensor;
      
      // Calculate targets using a simpler approach
      const targets = await tf.tidy(() => {
        const targetsArray: number[][] = [];
        
        batch.forEach((exp, index) => {
          const currentQ = currentQValues.slice([index, 0], [1, -1]).dataSync();
          const nextQ = nextQValues.slice([index, 0], [1, -1]).dataSync();
          
          const target = [...currentQ];
          const nextQValue = exp.done ? 0 : Math.max(...Array.from(nextQ));
          target[exp.action] = exp.reward + this.config.gamma * nextQValue;
          
          targetsArray.push(target);
        });
        
        return tf.tensor2d(targetsArray);
      });
      
      // Train the main network
      await this.mainNetwork.fit(statesTensor, targets, {
        epochs: 1,
        verbose: 0
      });
      
      // Clean up tensors
      statesTensor.dispose();
      nextStatesTensor.dispose();
      currentQValues.dispose();
      nextQValues.dispose();
      targets.dispose();
      
      // Update target network periodically
      this.updateCounter++;
      this.totalSteps++;
      
      if (this.updateCounter % this.config.targetUpdateFreq === 0) {
        await this.updateTargetNetwork();
      }
      
      // Decay epsilon
      this.config.epsilon = Math.max(
        this.config.epsilonMin,
        this.config.epsilon * this.config.epsilonDecay
      );
      
      this.isTraining = false;
      return { qValue: 0, epsilon: this.config.epsilon };
    } catch (error) {
      console.error('Error during DQN training:', error);
      this.isTraining = false;
      return { qValue: 0, epsilon: this.config.epsilon };
    }
  }

  private getValidActionsFromState(state: number[]): number[] {
    // Convert state back to board
    const board: number[][] = [];
    for (let i = 0; i < 9; i++) {
      board[i] = [];
      for (let j = 0; j < 9; j++) {
        board[i][j] = Math.round(state[i * 9 + j] * 9);
      }
    }
    return this.getValidActions(board);
  }

  private sampleBatch(batchSize: number): Experience[] {
    const batch: Experience[] = [];
    const size = Math.min(batchSize, this.replayBuffer.length);
    
    for (let i = 0; i < size; i++) {
      const randomIndex = Math.floor(Math.random() * this.replayBuffer.length);
      batch.push(this.replayBuffer[randomIndex]);
    }
    
    return batch;
  }

  private async updateTargetNetwork(): Promise<void> {
    if (!this.mainNetwork || !this.targetNetwork) return;
    
    const mainWeights = this.mainNetwork.getWeights();
    this.targetNetwork.setWeights(mainWeights);
  }

  // Get training statistics
  getTrainingStats(): { totalSteps: number; bufferSize: number; isTraining: boolean; epsilon: number } {
    return {
      totalSteps: this.totalSteps,
      bufferSize: this.replayBuffer.length,
      isTraining: this.isTraining,
      epsilon: this.config.epsilon
    };
  }

  // Export model weights
  async exportWeights(): Promise<{ main: any; target: any }> {
    if (!this.mainNetwork || !this.targetNetwork) {
      throw new Error('Models not initialized');
    }

    const mainWeights = this.mainNetwork.getWeights().map(w => w.arraySync());
    const targetWeights = this.targetNetwork.getWeights().map(w => w.arraySync());

    return {
      main: mainWeights,
      target: targetWeights
    };
  }

  // Import model weights
  async importWeights(weights: { main: any; target: any }): Promise<void> {
    if (!this.mainNetwork || !this.targetNetwork) {
      throw new Error('Models not initialized');
    }

    const mainTensors = weights.main.map((w: any) => tf.tensor(w));
    const targetTensors = weights.target.map((w: any) => tf.tensor(w));

    this.mainNetwork.setWeights(mainTensors);
    this.targetNetwork.setWeights(targetTensors);

    // Dispose tensors
    mainTensors.forEach((t: tf.Tensor) => t.dispose());
    targetTensors.forEach((t: tf.Tensor) => t.dispose());
  }

  // Update configuration
  updateConfig(newConfig: Partial<DQNConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  // Get current configuration
  getConfig(): DQNConfig {
    return { ...this.config };
  }

  // Save model
  async saveModel(path: string): Promise<void> {
    if (!this.mainNetwork || !this.targetNetwork) {
      throw new Error('Models not initialized');
    }

    await this.mainNetwork.save(`${path}/main`);
    await this.targetNetwork.save(`${path}/target`);
  }

  // Load model
  async loadModel(path: string): Promise<void> {
    this.mainNetwork = await tf.loadLayersModel(`${path}/main`);
    this.targetNetwork = await tf.loadLayersModel(`${path}/target`);
  }

  // Dispose of models
  dispose(): void {
    this.isTraining = false;
    if (this.mainNetwork) {
      this.mainNetwork.dispose();
    }
    if (this.targetNetwork) {
      this.targetNetwork.dispose();
    }
  }
}