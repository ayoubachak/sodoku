import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSliderModule } from '@angular/material/slider';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBar } from '@angular/material/snack-bar';
import { FormsModule } from '@angular/forms';
import { NgxChartsModule, Color } from '@swimlane/ngx-charts';

import { SudokuService } from '../../services/sudoku.service';
import { CellComponent } from '../cell/cell.component';
import { PPOAgentService, PPOConfig, TrainingStep } from '../../services/ppo-agent.service';
import { DQNAgentService, DQNConfig, Experience } from '../../services/dqn-agent.service';

interface ModelData {
  algorithm: 'ppo' | 'dqn';
  weights: any;
  configuration: {
    learningRate: number;
    batchSize: number;
    entropyCoef?: number;
    gamma?: number; // Use gamma for both PPO and DQN
    epsilon?: number;
    epsilonDecay?: number;
    targetUpdateFreq?: number;
  };
  stats: {
    accuracy: number;
    averageReward: number;
    episodes: number;
  };
  timestamp: number;
}

@Component({
  selector: 'app-ai-learning',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatCardModule,
    MatIconModule,
    MatSelectModule,
    MatProgressBarModule,
    MatSliderModule,
    FormsModule,
    NgxChartsModule,
    CellComponent
  ],
  templateUrl: './ai-learning.component.html',
  styleUrl: './ai-learning.component.css'
})
export class AiLearningComponent implements OnInit, OnDestroy {
  // Make Math available in template
  Math = Math;
  
  // Core properties
  loading = true;
  board: number[][] = [];
  solution: number[][] = [];
  selectedAlgorithm: 'ppo' | 'dqn' = 'ppo';
  
  // Training state
  isTraining = false;
  progress = 0;
  trainingSpeed = 50; // Default speed (0-100)
  episodes = 0;
  
  // Advanced settings
  showAdvancedSettings = false;
  learningRate = 0.0003;
  batchSize = 64;
  entropyCoef = 0.01;
  gamma = 0.99; // Changed from discountFactor to gamma to match DQN config
  epsilon = 1.0; // DQN exploration rate
  epsilonDecay = 0.995; // DQN epsilon decay
  targetUpdateFreq = 100; // DQN target network update frequency
  
  // Training stats
  currentReward = 0;
  averageReward = 0;
  accuracy = 0;
  actorLoss = 0;
  criticLoss = 0;
  avgAdvantage = 0;
  qValue = 0; // DQN Q-value
  explorationRate = 0; // DQN exploration rate
  
  // Episode tracking
  currentEpisodeSteps: TrainingStep[] = [];
  totalSteps = 0;
  
  // Testing stats
  showModelPerformance = false;
  testAccuracy = 0;
  testTime = 0;
  correctCells = 0;
  totalTestCells = 0;
  
  // Chart data for ngx-charts
  accuracyChartData: any[] = [];
  rewardChartData: any[] = [];
  
  // Chart configuration
  colorScheme: string | Color = 'cool';
  // curve: CurveFactory = curveMonotoneX;
  
  // Chart options
  showXAxis = true;
  showYAxis = true;
  gradient = false;
  showLegend = true;
  showXAxisLabel = true;
  xAxisLabel = 'Episode';
  showYAxisLabel = true;
  yAxisLabelAccuracy = 'Accuracy (%)';
  yAxisLabelReward = 'Reward';
  autoScale = true;
  
  // Pre-calculated random values for thinking process visualization
  thinkingProbabilities: number[] = [];
  thinkingQValues: number[] = [];

  // Visualizations
  cellHighlights: { row: number, col: number, value: number }[] = [];
  networkVisualization: any[] = [];
  
  // Board representation for the cell components
  cellData: {
    row: number;
    col: number;
    value: number;
    notes: number[];
    isOriginal: boolean;
    isSelected: boolean;
    isSameNumber: boolean;
    isHighlighted: boolean;
  }[][] = [];

  // File input reference
  @ViewChild('fileInput') fileInput!: ElementRef;

  // Training control
  private trainingInterval: any = null;
  private currentTrainingBoard: number[][] = [];

  // Neural network visualization properties
  hasNeuralNetwork: boolean = false;
  showNetworkVisualization: boolean = false;
  networkArchitecture: {name: string, units: number}[] = [];

  constructor(
    private readonly router: Router,
    private readonly sudokuService: SudokuService,
    private readonly snackBar: MatSnackBar,
    private readonly ppoAgent: PPOAgentService,
    private readonly dqnAgent: DQNAgentService
  ) {
    // Pre-generate random values for thinking visualization
    this.generateRandomThinkingValues();
    
    // Initialize chart data
    this.accuracyChartData = [{ name: 'Accuracy', series: [] }];
    this.rewardChartData = [{ name: 'Reward', series: [] }];
  }

  // Generate fixed random values to avoid ExpressionChangedAfterItHasBeenCheckedError
  private generateRandomThinkingValues(): void {
    // Generate 20 random probabilities between 0.5 and 1.0
    for (let i = 0; i < 20; i++) {
      this.thinkingProbabilities.push(0.5 + Math.random() * 0.5);
      this.thinkingQValues.push(-5 + Math.random() * 10); // Random Q-values between -5 and 5
    }
  }

  // Get a pre-generated probability value
  getThinkingProbability(index: number): number {
    return this.thinkingProbabilities[index % this.thinkingProbabilities.length];
  }

  // Get a pre-generated Q-value
  getThinkingQValue(index: number): number {
    return this.thinkingQValues[index % this.thinkingQValues.length];
  }

  async ngOnInit(): Promise<void> {
    // Initialize board
    this.initializeBoard();
    
    // Update network architecture first
    this.updateNetworkArchitecture();
    
    // Initialize PPO agent when PPO is selected
    if (this.selectedAlgorithm === 'ppo') {
      await this.initializePPOAgent();
    } else if (this.selectedAlgorithm === 'dqn') {
      await this.initializeDQNAgent();
    }
  }
  
  ngOnDestroy(): void {
    if (this.isTraining) {
      this.stopTraining();
    }
    
    // Dispose of agent resources
    this.ppoAgent.dispose();
    this.dqnAgent.dispose();
  }

  private async initializePPOAgent(): Promise<void> {
    const config: Partial<PPOConfig> = {
      learningRate: this.learningRate,
      batchSize: this.batchSize,
      entropyCoef: this.entropyCoef
    };
    
    await this.ppoAgent.initialize(config);
    console.log('PPO Agent initialized');
  }

  private async initializeDQNAgent(): Promise<void> {
    const config: Partial<DQNConfig> = {
      learningRate: this.learningRate,
      batchSize: this.batchSize,
      gamma: this.gamma, // Use gamma instead of discountFactor
      epsilon: this.epsilon,
      epsilonDecay: this.epsilonDecay,
      targetUpdateFreq: this.targetUpdateFreq
    };
    
    await this.dqnAgent.initialize(config);
    console.log('DQN Agent initialized');
  }

  private initializeBoard(): void {
    this.loading = true;
    this.sudokuService.getNewBoard().subscribe(boardData => {
      this.board = JSON.parse(JSON.stringify(boardData.grid));
      this.solution = JSON.parse(JSON.stringify(boardData.solution));
      this.currentTrainingBoard = JSON.parse(JSON.stringify(boardData.grid));
      this.initializeCellData();
      this.loading = false;
    });
  }

  private initializeCellData(): void {
    this.cellData = [];
    for (let row = 0; row < 9; row++) {
      this.cellData[row] = [];
      for (let col = 0; col < 9; col++) {
        this.cellData[row][col] = {
          row: row,
          col: col,
          value: this.board[row][col],
          notes: [],
          isOriginal: this.board[row][col] !== 0,
          isSelected: false,
          isSameNumber: false,
          isHighlighted: false
        };
      }
    }
  }

  getSections() {
    // Group cells into 3x3 sections for the grid display
    const sections = [];
    for (let sectionRow = 0; sectionRow < 3; sectionRow++) {
      for (let sectionCol = 0; sectionCol < 3; sectionCol++) {
        const section = [];
        
        for (let cellRow = 0; cellRow < 3; cellRow++) {
          for (let cellCol = 0; cellCol < 3; cellCol++) {
            const row = sectionRow * 3 + cellRow;
            const col = sectionCol * 3 + cellCol;
            
            section.push(this.cellData[row][col]);
          }
        }
        
        sections.push(section);
      }
    }
    
    return sections;
  }

  toggleAdvancedSettings(): void {
    this.showAdvancedSettings = !this.showAdvancedSettings;
  }

  async startTraining(): Promise<void> {
    if (this.isTraining) {
      this.stopTraining();
      return;
    }
    
    // Initialize PPO agent if not already done or if algorithm changed
    if (this.selectedAlgorithm === 'ppo') {
      await this.initializePPOAgent();
    } else if (this.selectedAlgorithm === 'dqn') {
      await this.initializeDQNAgent();
    }
    
    this.isTraining = true;
    this.progress = 0;
    this.episodes = 0;
    this.accuracy = 0;
    this.averageReward = 0;
    this.actorLoss = 0;
    this.criticLoss = 0;
    this.avgAdvantage = 0;
    this.totalSteps = 0;
    
    // Reset chart data
    this.accuracyChartData = [{ name: 'Accuracy', series: [] }];
    this.rewardChartData = [{ name: 'Reward', series: [] }];
    
    // Start training loop
    this.trainWithRealAlgorithm();
  }

  stopTraining(): void {
    this.isTraining = false;
    if (this.trainingInterval) {
      clearInterval(this.trainingInterval);
      this.trainingInterval = null;
    }
    // Clear any visualization data
    this.cellHighlights = [];
    this.resetHighlights();
  }

  resetHighlights(): void {
    // Clear all highlights
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        this.cellData[row][col].isHighlighted = false;
      }
    }
  }

  private async trainWithRealAlgorithm(): Promise<void> {
    if (this.selectedAlgorithm === 'ppo') {
      await this.trainPPO();
    } else {
      await this.trainDQN();
    }
  }

  private async trainPPO(): Promise<void> {
    const maxEpisodes = 1000;
    let episodeRewards: number[] = [];
    
    while (this.isTraining && this.episodes < maxEpisodes) {
      // Start new episode
      this.currentTrainingBoard = JSON.parse(JSON.stringify(this.board));
      this.currentEpisodeSteps = [];
      let episodeReward = 0;
      let stepCount = 0;
      const maxStepsPerEpisode = 100;
      
      // Run episode
      while (this.isTraining && stepCount < maxStepsPerEpisode) {
        try {
          // Get action from PPO agent
          const { action, logProb, value } = await this.ppoAgent.selectAction(this.currentTrainingBoard);
          
          // Apply action and get reward
          const result = this.ppoAgent.applyAction(this.currentTrainingBoard, action);
          
          // Store training step
          const trainingStep: TrainingStep = {
            state: this.boardToState(this.currentTrainingBoard),
            action: action,
            reward: result.reward,
            value: value,
            logProb: logProb,
            done: result.done
          };
          
          this.currentEpisodeSteps.push(trainingStep);
          this.ppoAgent.storeStep(trainingStep);
          
          // Update visualization
          this.updateVisualization(action, result.reward);
          
          // Update board
          this.currentTrainingBoard = result.newBoard;
          episodeReward += result.reward;
          stepCount++;
          this.totalSteps++;
          
          // Check if episode is done
          if (result.done) {
            this.accuracy = 100; // Solved the puzzle
            break;
          }
          
          // Check if no valid moves left
          const validActions = this.getValidActions(this.currentTrainingBoard);
          if (validActions.length === 0) {
            break;
          }
          
        } catch (error) {
          console.error('Error during PPO training step:', error);
          break;
        }
        
        // Small delay for visualization
        await new Promise(resolve => setTimeout(resolve, Math.max(10, 1000 - this.trainingSpeed * 10)));
      }
      
      // Episode finished
      episodeRewards.push(episodeReward);
      this.episodes++;
      
      // Calculate running averages
      this.currentReward = episodeReward;
      this.averageReward = episodeRewards.reduce((a, b) => a + b, 0) / episodeRewards.length;
      
      // Calculate accuracy (percentage of cells correctly filled)
      this.accuracy = this.calculateAccuracy();
      
      // Train the PPO agent
      if (this.ppoAgent.getTrainingStats().bufferSize >= this.batchSize) {
        const trainingResult = await this.ppoAgent.train();
        this.actorLoss = trainingResult.actorLoss;
        this.criticLoss = trainingResult.criticLoss;
        this.avgAdvantage = trainingResult.avgAdvantage;
      }
      
      // Update progress
      this.progress = Math.min(100, (this.episodes / maxEpisodes) * 100);
      
      // Update charts every 5 episodes
      if (this.episodes % 5 === 0) {
        this.updateCharts();
      }
      
      // Reset board for next episode
      if (this.isTraining) {
        this.initializeBoard();
        await new Promise(resolve => setTimeout(resolve, 100)); // Wait for board initialization
      }
    }
    
    // Training complete
    if (this.isTraining) {
      this.progress = 100;
      this.isTraining = false;
      this.snackBar.open('PPO Training complete!', 'Close', { duration: 3000 });
    }
  }

  private async trainDQN(): Promise<void> {
    const maxEpisodes = 1000;
    let episodeRewards: number[] = [];
    
    while (this.isTraining && this.episodes < maxEpisodes) {
      // Start new episode
      this.currentTrainingBoard = JSON.parse(JSON.stringify(this.board));
      let episodeReward = 0;
      let stepCount = 0;
      const maxStepsPerEpisode = 100;
      
      // Run episode
      while (this.isTraining && stepCount < maxStepsPerEpisode) {
        try {
          // Get action from DQN agent
          const { action, qValue } = await this.dqnAgent.selectAction(this.currentTrainingBoard);
          
          // Apply action and get reward
          const result = this.dqnAgent.applyAction(this.currentTrainingBoard, action);
          
          // Store experience in replay buffer
          const experience: Experience = {
            state: this.boardToState(this.currentTrainingBoard),
            action: action,
            reward: result.reward,
            nextState: this.boardToState(result.newBoard),
            done: result.done
          };
          
          this.dqnAgent.storeExperience(experience);
          
          // Update visualization
          this.updateVisualization(action, result.reward);
          this.qValue = qValue;
          
          // Update board
          this.currentTrainingBoard = result.newBoard;
          episodeReward += result.reward;
          stepCount++;
          this.totalSteps++;
          
          // Check if episode is done
          if (result.done) {
            this.accuracy = 100; // Solved the puzzle
            break;
          }
          
          // Check if no valid moves left
          const validActions = this.getValidActions(this.currentTrainingBoard);
          if (validActions.length === 0) {
            break;
          }
          
        } catch (error) {
          console.error('Error during DQN training step:', error);
          break;
        }
        
        // Small delay for visualization
        await new Promise(resolve => setTimeout(resolve, Math.max(10, 1000 - this.trainingSpeed * 10)));
      }
      
      // Episode finished
      episodeRewards.push(episodeReward);
      this.episodes++;
      
      // Calculate running averages
      this.currentReward = episodeReward;
      this.averageReward = episodeRewards.reduce((a, b) => a + b, 0) / episodeRewards.length;
      
      // Calculate accuracy (percentage of cells correctly filled)
      this.accuracy = this.calculateAccuracy();
      
      // Train the DQN agent
      if (this.dqnAgent.getTrainingStats().bufferSize >= this.batchSize) {
        const trainingResult = await this.dqnAgent.train();
        this.qValue = trainingResult.qValue;
        this.explorationRate = trainingResult.epsilon;
      }
      
      // Update progress
      this.progress = Math.min(100, (this.episodes / maxEpisodes) * 100);
      
      // Update charts every 5 episodes
      if (this.episodes % 5 === 0) {
        this.updateCharts();
      }
      
      // Reset board for next episode
      if (this.isTraining) {
        this.initializeBoard();
        await new Promise(resolve => setTimeout(resolve, 100)); // Wait for board initialization
      }
    }
    
    // Training complete
    if (this.isTraining) {
      this.progress = 100;
      this.isTraining = false;
      this.snackBar.open('DQN Training complete!', 'Close', { duration: 3000 });
    }
  }

  private boardToState(board: number[][]): number[] {
    const state: number[] = [];
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        state.push(board[i][j] / 9.0); // Normalize to [0, 1]
      }
    }
    return state;
  }

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

  private updateVisualization(action: number, reward: number): void {
    const row = Math.floor(action / 81);
    const col = Math.floor((action % 81) / 9);
    const value = (action % 9) + 1;
    
    // Clear previous highlights
    this.resetHighlights();
    
    // Highlight the cell being modified
    this.cellData[row][col].isHighlighted = true;
    this.cellData[row][col].value = value;
    
    // Add to highlights array for template
    this.cellHighlights = [{ row, col, value }];
  }

  private calculateAccuracy(): number {
    let correctCells = 0;
    let totalFilledCells = 0;
    
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (this.currentTrainingBoard[row][col] !== 0) {
          totalFilledCells++;
          if (this.currentTrainingBoard[row][col] === this.solution[row][col]) {
            correctCells++;
          }
        }
      }
    }
    
    return totalFilledCells > 0 ? (correctCells / totalFilledCells) * 100 : 0;
  }

  // Fallback training method for DQN or when PPO fails
  trainWithAlgorithm(): void {
    // This is a simulation of training
    // In a real implementation, you would use TensorFlow.js here
    
    const trainingStep = () => {
      if (!this.isTraining) return;
      
      this.episodes++;
      this.progress = Math.min(100, this.episodes * 2);
      
      // Simulate learning progress
      this.accuracy = Math.min(95, this.episodes * 0.8 + Math.random() * 5);
      this.currentReward = -50 + this.episodes * 0.5 + (Math.random() - 0.5) * 10;
      this.averageReward = this.currentReward * 0.9;
      
      // Simulate some network visualization updates
      this.generateNetworkVisualization();
      
      // Update charts every 5 episodes
      if (this.episodes % 5 === 0) {
        this.updateCharts();
      }
      
      // Continue training until reaching 100%
      if (this.progress < 100) {
        const speed = 1000 - (this.trainingSpeed * 9); // Convert 0-100 to 1000-100ms
        setTimeout(trainingStep, speed);
      } else {
        // Training complete
        this.progress = 100;
        this.isTraining = false;
        this.snackBar.open('Training complete!', 'Close', { duration: 3000 });
      }
    };
    
    // Start training loop
    trainingStep();
  }
  
  updateCharts(): void {
    // Add new data points to the charts
    this.accuracyChartData[0].series.push({
      name: this.episodes.toString(),
      value: this.accuracy
    });
    
    this.rewardChartData[0].series.push({
      name: this.episodes.toString(),
      value: this.averageReward
    });
    
    // Limit chart data points for performance
    const maxDataPoints = 30;
    if (this.accuracyChartData[0].series.length > maxDataPoints) {
      this.accuracyChartData[0].series = this.accuracyChartData[0].series.slice(-maxDataPoints);
      this.rewardChartData[0].series = this.rewardChartData[0].series.slice(-maxDataPoints);
    }
    
    // Create new array references to trigger Angular change detection
    this.accuracyChartData = [...this.accuracyChartData];
    this.rewardChartData = [...this.rewardChartData];
  }

  generateNetworkVisualization(): void {
    // This is a simplified mock visualization
    // In a real implementation, you would visualize actual network weights/activations
    
    // Simple array of "activation" values for visualization
    const layerSizes = this.selectedAlgorithm === 'ppo' 
      ? [81, 256, 128, 64, 729] // Updated to match PPO architecture
      : [81, 128, 64, 729];
      
    this.networkVisualization = [];
    
    for (let layer = 0; layer < layerSizes.length; layer++) {
      const layerData = [];
      for (let i = 0; i < layerSizes[layer]; i++) {
        // Only include a subset of nodes for visualization
        if (i < 10 || i > layerSizes[layer] - 5 || i % Math.ceil(layerSizes[layer] / 10) === 0) {
          layerData.push({
            id: `${layer}_${i}`,
            value: Math.random(),
            size: layer === 0 || layer === layerSizes.length - 1 ? 'small' : 'medium'
          });
        }
      }
      
      this.networkVisualization.push(layerData);
    }
  }

  async testModel(): Promise<void> {
    if (this.selectedAlgorithm === 'ppo') {
      await this.testPPOModel();
    } else if (this.selectedAlgorithm === 'dqn') {
      await this.testDQNModel();
    } else {
      this.testSimulatedModel();
    }
  }

  private async testPPOModel(): Promise<void> {
    // Reset highlights
    this.resetHighlights();
    
    // Test the PPO model on current board
    const testBoard = JSON.parse(JSON.stringify(this.board));
    let step = 0;
    const maxSteps = 50;
    
    try {
      while (step < maxSteps) {
        const validActions = this.getValidActions(testBoard);
        if (validActions.length === 0) break;
        
        const { action } = await this.ppoAgent.selectAction(testBoard);
        const result = this.ppoAgent.applyAction(testBoard, action);
        
        // Update visualization
        this.updateVisualization(action, result.reward);
        
        testBoard.splice(0, testBoard.length, ...result.newBoard);
        step++;
        
        if (result.done) {
          this.snackBar.open('PPO Agent solved the puzzle!', 'Close', { duration: 3000 });
          break;
        }
        
        // Delay for visualization
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    } catch (error) {
      console.error('Error testing PPO model:', error);
      this.snackBar.open('Error testing model', 'Close', { duration: 3000 });
    }
  }

  private async testDQNModel(): Promise<void> {
    // Reset highlights
    this.resetHighlights();
    
    // Test the DQN model on current board
    const testBoard = JSON.parse(JSON.stringify(this.board));
    let step = 0;
    const maxSteps = 50;
    
    try {
      while (step < maxSteps) {
        const validActions = this.getValidActions(testBoard);
        if (validActions.length === 0) break;
        
        const { action, qValue } = await this.dqnAgent.selectAction(testBoard);
        const result = this.dqnAgent.applyAction(testBoard, action);
        
        // Update visualization
        this.updateVisualization(action, result.reward);
        this.qValue = qValue;
        
        testBoard.splice(0, testBoard.length, ...result.newBoard);
        step++;
        
        if (result.done) {
          this.snackBar.open('DQN Agent solved the puzzle!', 'Close', { duration: 3000 });
          break;
        }
        
        // Delay for visualization
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    } catch (error) {
      console.error('Error testing DQN model:', error);
      this.snackBar.open('Error testing model', 'Close', { duration: 3000 });
    }
  }

  private testSimulatedModel(): void {
    // Reset highlights
    this.resetHighlights();
    
    // Simulate testing the model
    let step = 0;
    const maxSteps = 10;
    
    const testStep = () => {
      if (step >= maxSteps) {
        this.snackBar.open('Simulated model test complete!', 'Close', { duration: 3000 });
        return;
      }
      
      // Simulate making a move
      const emptyCells = [];
      for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
          if (this.cellData[row][col].value === 0) {
            emptyCells.push({ row, col });
          }
        }
      }
      
      if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        const randomValue = Math.floor(Math.random() * 9) + 1;
        
        // Update visualization
        this.resetHighlights();
        this.cellData[randomCell.row][randomCell.col].isHighlighted = true;
        this.cellData[randomCell.row][randomCell.col].value = randomValue;
        this.cellHighlights = [{ row: randomCell.row, col: randomCell.col, value: randomValue }];
      }
      
      step++;
      setTimeout(testStep, 500);
    };
    
    testStep();
  }

  async saveModel(): Promise<void> {
    if (this.selectedAlgorithm === 'ppo') {
      await this.savePPOModel();
    } else if (this.selectedAlgorithm === 'dqn') {
      await this.saveDQNModel();
    } else {
      this.saveSimulatedModel();
    }
  }

  private async savePPOModel(): Promise<void> {
    try {
      const weights = await this.ppoAgent.exportWeights();
      const config = this.ppoAgent.getConfig();
      
      const modelData: ModelData = {
        algorithm: this.selectedAlgorithm,
        weights: weights,
        configuration: {
          learningRate: config.learningRate,
          batchSize: config.batchSize,
          entropyCoef: config.entropyCoef
        },
        stats: {
          accuracy: this.accuracy,
          averageReward: this.averageReward,
          episodes: this.episodes
        },
        timestamp: Date.now()
      };
      
      localStorage.setItem('sudoku_ai_model', JSON.stringify(modelData));
      
      this.snackBar.open('PPO Model saved successfully!', 'Close', {
        duration: 3000
      });
    } catch (error) {
      console.error('Error saving PPO model:', error);
      this.snackBar.open('Error saving model', 'Close', { duration: 3000 });
    }
  }

  private async saveDQNModel(): Promise<void> {
    try {
      const weights = await this.dqnAgent.exportWeights();
      const config = this.dqnAgent.getConfig();
      
      const modelData: ModelData = {
        algorithm: this.selectedAlgorithm,
        weights: weights,
        configuration: {
          learningRate: config.learningRate,
          batchSize: config.batchSize,
          gamma: config.gamma,
          epsilon: config.epsilon,
          epsilonDecay: config.epsilonDecay,
          targetUpdateFreq: config.targetUpdateFreq
        },
        stats: {
          accuracy: this.accuracy,
          averageReward: this.averageReward,
          episodes: this.episodes
        },
        timestamp: Date.now()
      };
      
      localStorage.setItem('sudoku_ai_model', JSON.stringify(modelData));
      
      this.snackBar.open('DQN Model saved successfully!', 'Close', {
        duration: 3000
      });
    } catch (error) {
      console.error('Error saving DQN model:', error);
      this.snackBar.open('Error saving model', 'Close', { duration: 3000 });
    }
  }

  private saveSimulatedModel(): void {
    // Simulate saving the model to localStorage
    const modelData: ModelData = {
      algorithm: this.selectedAlgorithm,
      weights: { /* Would contain actual model weights */ },
      configuration: {
        learningRate: this.learningRate,
        batchSize: this.batchSize,
        ...(this.selectedAlgorithm === 'ppo' ? { entropyCoef: this.entropyCoef } : { gamma: this.gamma })
      },
      stats: {
        accuracy: this.accuracy,
        averageReward: this.averageReward,
        episodes: this.episodes
      },
      timestamp: Date.now()
    };
    
    localStorage.setItem('sudoku_ai_model', JSON.stringify(modelData));
    
    this.snackBar.open('Model saved successfully!', 'Close', {
      duration: 3000
    });
  }
  
  async loadModel(): Promise<void> {
    if (this.selectedAlgorithm === 'ppo') {
      await this.loadPPOModel();
    } else if (this.selectedAlgorithm === 'dqn') {
      await this.loadDQNModel();
    } else {
      this.loadSimulatedModel();
    }
  }

  private async loadPPOModel(): Promise<void> {
    try {
      const savedModel = localStorage.getItem('sudoku_ai_model');
      
      if (savedModel) {
        const modelData: ModelData = JSON.parse(savedModel);
        
        if (modelData.algorithm === 'ppo' && modelData.weights) {
          // Initialize PPO agent first
          await this.initializePPOAgent();
          
          // Load weights
          await this.ppoAgent.importWeights(modelData.weights);
          
          // Apply saved settings
          this.learningRate = modelData.configuration.learningRate;
          this.batchSize = modelData.configuration.batchSize;
          if (modelData.configuration.entropyCoef) {
            this.entropyCoef = modelData.configuration.entropyCoef;
          }
          
          // Apply saved stats
          this.accuracy = modelData.stats.accuracy;
          this.averageReward = modelData.stats.averageReward;
          this.episodes = modelData.stats.episodes;
          this.progress = 100; // Assume a loaded model is fully trained
          
          // Update charts
          this.updateChartsFromLoadedModel();
          
          this.snackBar.open('PPO Model loaded successfully!', 'Close', {
            duration: 3000
          });
        } else {
          throw new Error('Invalid PPO model data');
        }
      } else {
        this.snackBar.open('No saved PPO model found!', 'Close', {
          duration: 3000
        });
      }
    } catch (error) {
      console.error('Error loading PPO model:', error);
      this.snackBar.open('Failed to load model', 'Close', { duration: 3000 });
    }
  }

  private async loadDQNModel(): Promise<void> {
    try {
      const savedModel = localStorage.getItem('sudoku_ai_model');
      
      if (savedModel) {
        const modelData: ModelData = JSON.parse(savedModel);
        
        if (modelData.algorithm === 'dqn' && modelData.weights) {
          // Initialize DQN agent first
          await this.initializeDQNAgent();
          
          // Load weights
          await this.dqnAgent.importWeights(modelData.weights);
          
          // Apply saved settings
          this.learningRate = modelData.configuration.learningRate;
          this.batchSize = modelData.configuration.batchSize;
          if (modelData.configuration.gamma) {
            this.gamma = modelData.configuration.gamma;
          }
          if (modelData.configuration.epsilon) {
            this.epsilon = modelData.configuration.epsilon;
          }
          if (modelData.configuration.epsilonDecay) {
            this.epsilonDecay = modelData.configuration.epsilonDecay;
          }
          if (modelData.configuration.targetUpdateFreq) {
            this.targetUpdateFreq = modelData.configuration.targetUpdateFreq;
          }
          
          // Apply saved stats
          this.accuracy = modelData.stats.accuracy;
          this.averageReward = modelData.stats.averageReward;
          this.episodes = modelData.stats.episodes;
          this.progress = 100; // Assume a loaded model is fully trained
          
          // Update charts
          this.updateChartsFromLoadedModel();
          
          this.snackBar.open('DQN Model loaded successfully!', 'Close', {
            duration: 3000
          });
        } else {
          throw new Error('Invalid DQN model data');
        }
      } else {
        this.snackBar.open('No saved DQN model found!', 'Close', {
          duration: 3000
        });
      }
    } catch (error) {
      console.error('Error loading DQN model:', error);
      this.snackBar.open('Failed to load model', 'Close', { duration: 3000 });
    }
  }

  private loadSimulatedModel(): void {
    try {
      const savedModel = localStorage.getItem('sudoku_ai_model');
      
      if (savedModel) {
        const modelData: ModelData = JSON.parse(savedModel);
        
        // Apply saved settings
        this.learningRate = modelData.configuration.learningRate;
        this.batchSize = modelData.configuration.batchSize;
        if (modelData.configuration.entropyCoef) {
          this.entropyCoef = modelData.configuration.entropyCoef;
        }
        if (modelData.configuration.gamma) {
          this.gamma = modelData.configuration.gamma;
        }
        
        // Apply saved stats
        this.accuracy = modelData.stats.accuracy;
        this.averageReward = modelData.stats.averageReward;
        this.episodes = modelData.stats.episodes;
        this.progress = 100;
        
        this.updateChartsFromLoadedModel();
        
        this.snackBar.open('Model loaded successfully!', 'Close', {
          duration: 3000
        });
      } else {
        this.snackBar.open('No saved model found!', 'Close', {
          duration: 3000
        });
      }
    } catch (error) {
      console.error('Error loading model:', error);
      this.snackBar.open('Failed to load model', 'Close', { duration: 3000 });
    }
  }

  generateNewSudoku(): void {
    // Generate new puzzle
    this.initializeBoard();
    
    // Reset any highlights
    this.resetHighlights();
    
    this.snackBar.open('New puzzle generated!', 'Close', { duration: 2000 });
  }

  testOnCurrentBoard(): void {
    // Test the current model on the current board
    this.testModel();
  }

  // Get visible nodes for network visualization (limit for display)
  getVisibleNodes(totalNodes: number): number[] {
    const maxVisible = 5;
    if (totalNodes <= maxVisible) {
      return Array.from({ length: totalNodes }, (_, i) => i);
    } else {
      return Array.from({ length: maxVisible }, (_, i) => i);
    }
  }

  // Helper method to handle cell clicks in the Sudoku board
  onCellClick(row: number, col: number): void {
    // For visualization purposes only - not interactive during training
    if (!this.isTraining) {
      console.log(`Cell clicked: (${row}, ${col})`);
    }
  }

  // Handle algorithm selection change
  async onAlgorithmChange(): Promise<void> {
    if (this.isTraining) {
      this.stopTraining();
    }

    // Initialize the selected algorithm
    if (this.selectedAlgorithm === 'ppo') {
      await this.initializePPOAgent();
    } else if (this.selectedAlgorithm === 'dqn') {
      await this.initializeDQNAgent();
    }

    // Update network architecture visualization
    this.updateNetworkArchitecture();

    // Reset training stats
    this.resetTrainingStats();

    this.snackBar.open(`Switched to ${this.selectedAlgorithm.toUpperCase()} algorithm`, 'Close', { duration: 2000 });
  }

  private resetTrainingStats(): void {
    this.progress = 0;
    this.episodes = 0;
    this.accuracy = 0;
    this.averageReward = 0;
    this.currentReward = 0;
    this.actorLoss = 0;
    this.criticLoss = 0;
    this.avgAdvantage = 0;
    this.qValue = 0;
    this.explorationRate = 0;
    this.totalSteps = 0;
    
    // Reset chart data
    this.accuracyChartData = [{ name: 'Accuracy', series: [] }];
    this.rewardChartData = [{ name: 'Reward', series: [] }];
  }

  updateNetworkArchitecture(): void {
    if (this.selectedAlgorithm === 'ppo') {
      this.networkArchitecture = [
        { name: 'Input', units: 81 },  // Sudoku board state
        { name: 'Hidden 1', units: 256 },
        { name: 'Hidden 2', units: 128 },
        { name: 'Hidden 3', units: 64 },
        { name: 'Policy/Value', units: 729 }   // Policy output (729 actions)
      ];
      this.hasNeuralNetwork = true;
    } else if (this.selectedAlgorithm === 'dqn') {
      this.networkArchitecture = [
        { name: 'Input', units: 81 },  // Sudoku board state
        { name: 'Shared', units: 128 },
        { name: 'Value Stream', units: 64 },
        { name: 'Advantage Stream', units: 64 },
        { name: 'Q-Values', units: 729 } // 729 Q-values (81 cells × 9 values)
      ];
      this.hasNeuralNetwork = true;
    } else {
      this.networkArchitecture = [];
      this.hasNeuralNetwork = false;
    }
    this.showNetworkVisualization = this.hasNeuralNetwork;
  }

  // Update configuration when settings change
  async updatePPOConfig(): Promise<void> {
    if (this.selectedAlgorithm === 'ppo') {
      const config: Partial<PPOConfig> = {
        learningRate: this.learningRate,
        batchSize: this.batchSize,
        entropyCoef: this.entropyCoef
      };
      
      this.ppoAgent.updateConfig(config);
    }
  }

  // Update DQN configuration when settings change
  async updateDQNConfig(): Promise<void> {
    if (this.selectedAlgorithm === 'dqn') {
      const config: Partial<DQNConfig> = {
        learningRate: this.learningRate,
        batchSize: this.batchSize,
        gamma: this.gamma,
        epsilon: this.epsilon,
        epsilonDecay: this.epsilonDecay,
        targetUpdateFreq: this.targetUpdateFreq
      };
      
      this.dqnAgent.updateConfig(config);
    }
  }

  // Update charts from loaded model data
  private updateChartsFromLoadedModel(): void {
    // Simulate chart data for loaded model
    const dataPoints = Math.min(10, this.episodes);
    
    // Generate simulated progress data
    for (let i = 1; i <= dataPoints; i++) {
      const episode = Math.floor((this.episodes / dataPoints) * i);
      const progressRatio = i / dataPoints;
      
      this.accuracyChartData[0].series.push({
        name: episode.toString(),
        value: this.accuracy * progressRatio
      });
      
      this.rewardChartData[0].series.push({
        name: episode.toString(),
        value: this.averageReward * progressRatio
      });
    }
    
    // Trigger chart update
    this.accuracyChartData = [...this.accuracyChartData];
    this.rewardChartData = [...this.rewardChartData];
    
  }

  // File operations
  triggerFileInput(): void {
    this.fileInput.nativeElement.click();
  }

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file && file.type === 'application/json') {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const modelData: ModelData = JSON.parse(e.target?.result as string);
          this.importModelFromFile(modelData);
        } catch (error) {
          console.error('Error parsing model file:', error);
          this.snackBar.open('Invalid model file format', 'Close', { duration: 3000 });
        }
      };
      reader.readAsText(file);
    } else {
      this.snackBar.open('Please select a valid JSON file', 'Close', { duration: 3000 });
    }
    
    // Reset file input
    event.target.value = '';
  }

  async importModelFromFile(modelData: ModelData): Promise<void> {
    try {
      if (modelData.algorithm === 'ppo') {
        // Initialize PPO agent first
        await this.initializePPOAgent();
        
        // Load weights
        await this.ppoAgent.importWeights(modelData.weights);
        
        // Apply settings
        this.selectedAlgorithm = modelData.algorithm;
        this.learningRate = modelData.configuration.learningRate;
        this.batchSize = modelData.configuration.batchSize;
        if (modelData.configuration.entropyCoef) {
          this.entropyCoef = modelData.configuration.entropyCoef;
        }
        
        // Apply stats
        this.accuracy = modelData.stats.accuracy;
        this.averageReward = modelData.stats.averageReward;
        this.episodes = modelData.stats.episodes;
        this.progress = 100;
        
        this.updateChartsFromLoadedModel();
        
        this.snackBar.open('PPO Model imported successfully!', 'Close', {
          duration: 3000
        });
      } else if (modelData.algorithm === 'dqn') {
        // Initialize DQN agent first
        await this.initializeDQNAgent();
        
        // Load weights
        await this.dqnAgent.importWeights(modelData.weights);
        
        // Apply settings
        this.selectedAlgorithm = modelData.algorithm;
        this.learningRate = modelData.configuration.learningRate;
        this.batchSize = modelData.configuration.batchSize;
        if (modelData.configuration.gamma) {
          this.gamma = modelData.configuration.gamma;
        }
        if (modelData.configuration.epsilon) {
          this.epsilon = modelData.configuration.epsilon;
        }
        if (modelData.configuration.epsilonDecay) {
          this.epsilonDecay = modelData.configuration.epsilonDecay;
        }
        if (modelData.configuration.targetUpdateFreq) {
          this.targetUpdateFreq = modelData.configuration.targetUpdateFreq;
        }
        
        // Apply stats
        this.accuracy = modelData.stats.accuracy;
        this.averageReward = modelData.stats.averageReward;
        this.episodes = modelData.stats.episodes;
        this.progress = 100;
        
        this.updateChartsFromLoadedModel();
        
        this.snackBar.open('DQN Model imported successfully!', 'Close', {
          duration: 3000
        });
      } else {
        this.snackBar.open('Unsupported model format', 'Close', { duration: 3000 });
      }
      
      // Update network architecture
      this.updateNetworkArchitecture();
      
    } catch (error) {
      console.error('Error importing model:', error);
      this.snackBar.open('Failed to import model', 'Close', { duration: 3000 });
    }
  }

  async exportModel(): Promise<void> {
    try {
      if (this.selectedAlgorithm === 'ppo') {
        const weights = await this.ppoAgent.exportWeights();
        const config = this.ppoAgent.getConfig();
        
        const modelData: ModelData = {
          algorithm: this.selectedAlgorithm,
          weights: weights,
          configuration: {
            learningRate: config.learningRate,
            batchSize: config.batchSize,
            entropyCoef: config.entropyCoef
          },
          stats: {
            accuracy: this.accuracy,
            averageReward: this.averageReward,
            episodes: this.episodes
          },
          timestamp: Date.now()
        };
        
        // Create and download file
        const blob = new Blob([JSON.stringify(modelData, null, 2)], {
          type: 'application/json'
        });
        
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `sudoku-ppo-model-${Date.now()}.json`;
        link.click();
        
        window.URL.revokeObjectURL(url);
        
        this.snackBar.open('PPO Model exported successfully!', 'Close', {
          duration: 3000
        });
      } else if (this.selectedAlgorithm === 'dqn') {
        const weights = await this.dqnAgent.exportWeights();
        const config = this.dqnAgent.getConfig();
        
        const modelData: ModelData = {
          algorithm: this.selectedAlgorithm,
          weights: weights,
          configuration: {
            learningRate: config.learningRate,
            batchSize: config.batchSize,
            gamma: config.gamma,
            epsilon: config.epsilon,
            epsilonDecay: config.epsilonDecay,
            targetUpdateFreq: config.targetUpdateFreq
          },
          stats: {
            accuracy: this.accuracy,
            averageReward: this.averageReward,
            episodes: this.episodes
          },
          timestamp: Date.now()
        };
        
        // Create and download file
        const blob = new Blob([JSON.stringify(modelData, null, 2)], {
          type: 'application/json'
        });
        
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `sudoku-dqn-model-${Date.now()}.json`;
        link.click();
        
        window.URL.revokeObjectURL(url);
        
        this.snackBar.open('DQN Model exported successfully!', 'Close', {
          duration: 3000
        });
      } else {
        this.snackBar.open('No trained model to export', 'Close', { duration: 3000 });
      }
    } catch (error) {
      console.error('Error exporting model:', error);
      this.snackBar.open('Failed to export model', 'Close', { duration: 3000 });
    }
  }

  newBoard(): void {
    this.generateNewSudoku();
  }

  backToMenu(): void {
    // Clean up any training processes
    if (this.isTraining) {
      this.stopTraining();
    }
    
    // Dispose of agent resources
    this.ppoAgent.dispose();
    this.dqnAgent.dispose();
    
    this.router.navigate(['/menu']);
  }
}