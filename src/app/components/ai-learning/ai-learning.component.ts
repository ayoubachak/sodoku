import { Component, OnInit, OnDestroy, AfterViewInit, ChangeDetectorRef, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSliderModule } from '@angular/material/slider';
import { MatSnackBar } from '@angular/material/snack-bar';
import { FormsModule } from '@angular/forms';
import { CellComponent } from '../cell/cell.component';
import { SudokuService } from '../../services/sudoku.service';
import { PPOAgentService, PPOConfig, TrainingStep } from '../../services/ppo-agent.service';

// Import ngx-charts modules, remove unsupported imports
import { NgxChartsModule, Color, ScaleType } from '@swimlane/ngx-charts';

interface ModelData {
  algorithm: 'ppo' | 'dqn';
  weights: any;
  configuration: {
    learningRate: number;
    batchSize: number;
    entropyCoef?: number;
    discountFactor?: number;
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
export class AiLearningComponent implements OnInit, OnDestroy, AfterViewInit {
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
  discountFactor = 0.99;
  
  // Training stats
  currentReward = 0;
  averageReward = 0;
  accuracy = 0;
  actorLoss = 0;
  criticLoss = 0;
  avgAdvantage = 0;
  
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
    private router: Router,
    private sudokuService: SudokuService,
    private snackBar: MatSnackBar,
    private cdr: ChangeDetectorRef,
    private ppoAgent: PPOAgentService
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
    
    // Initialize PPO agent when PPO is selected
    if (this.selectedAlgorithm === 'ppo') {
      await this.initializePPOAgent();
    }
  }
  
  ngAfterViewInit(): void {
    // Create charts with a slight delay to ensure DOM elements are ready
    setTimeout(() => {
      this.cdr.detectChanges();
    }, 100);
  }
  
  ngOnDestroy(): void {
    if (this.isTraining) {
      this.stopTraining();
    }
    
    // Dispose of PPO agent resources
    this.ppoAgent.dispose();
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

  private initializeBoard(): void {
    this.loading = true;
    this.sudokuService.getNewBoard().subscribe(boardData => {
      this.board = JSON.parse(JSON.stringify(boardData.grid));
      this.solution = JSON.parse(JSON.stringify(boardData.solution));
      this.currentTrainingBoard = JSON.parse(JSON.stringify(boardData.grid));
      this.initializeCellData();
      this.loading = false;
      this.cdr.detectChanges();
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
      // Fallback to simulation for DQN (to be implemented later)
      this.trainWithAlgorithm();
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
    
    this.cdr.detectChanges();
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
      : [81, 64, 32, 729];
      
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

  private testSimulatedModel(): void {
    // Simulate testing the model on the current board
    this.resetHighlights();
    
    // Highlight a sequence of moves to simulate the model solving the puzzle
    let emptyCells = [];
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (this.board[row][col] === 0) {
          emptyCells.push({ row, col, value: this.solution[row][col] });
        }
      }
    }
    
    // Shuffle the empty cells to simulate non-linear solving
    emptyCells = this.shuffleArray(emptyCells);
    
    let cellIndex = 0;
    const solveStep = () => {
      if (cellIndex < emptyCells.length) {
        const { row, col, value } = emptyCells[cellIndex];
        
        // Clear previous highlights
        this.resetHighlights();
        
        // Highlight current cell
        this.cellData[row][col].isHighlighted = true;
        this.cellData[row][col].value = value;
        this.cellHighlights = [{ row, col, value }];
        
        cellIndex++;
        
        if (cellIndex < emptyCells.length) {
          setTimeout(solveStep, 200);
        }
      }
    };
    
    solveStep();
  }
  
  generateNewSudoku(): void {
    this.initializeBoard();
  }
  
  testOnCurrentBoard(): void {
    // Simulate testing the trained model on the current board
    this.showModelPerformance = true;
    this.resetHighlights();
    
    // Start with all empty cells
    this.totalTestCells = 0;
    this.correctCells = 0;
    
    let emptyCells = [];
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (this.board[row][col] === 0) {
          emptyCells.push({ row, col, value: this.solution[row][col] });
          this.totalTestCells++;
        }
      }
    }
    
    // Simulate the actual testing
    const startTime = performance.now();
    
    // Shuffle the empty cells
    emptyCells = this.shuffleArray(emptyCells);
    
    // Process cells with a delay to visualize the process
    let processedCells = 0;
    const processCell = () => {
      if (processedCells < emptyCells.length) {
        const { row, col, value } = emptyCells[processedCells];
        
        // Simulate model prediction (80% accuracy)
        const isCorrect = Math.random() < 0.8;
        if (isCorrect) {
          this.correctCells++;
          this.cellData[row][col].value = value;
        } else {
          // Wrong prediction
          this.cellData[row][col].value = Math.floor(Math.random() * 9) + 1;
        }
        
        this.cellData[row][col].isHighlighted = true;
        
        processedCells++;
        this.testAccuracy = (this.correctCells / processedCells) * 100;
        
        setTimeout(() => {
          this.cellData[row][col].isHighlighted = false;
          if (processedCells < emptyCells.length) {
            setTimeout(processCell, 100);
          } else {
            // Testing complete
            this.testTime = performance.now() - startTime;
            this.snackBar.open(`Testing complete! Accuracy: ${this.testAccuracy.toFixed(1)}%`, 'Close', {
              duration: 5000
            });
          }
        }, 300);
      }
    };
    
    processCell();
  }
  
  async saveModel(): Promise<void> {
    if (this.selectedAlgorithm === 'ppo') {
      await this.savePPOModel();
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

  private saveSimulatedModel(): void {
    // Simulate saving the model to localStorage
    const modelData: ModelData = {
      algorithm: this.selectedAlgorithm,
      weights: { /* Would contain actual model weights */ },
      configuration: {
        learningRate: this.learningRate,
        batchSize: this.batchSize,
        ...(this.selectedAlgorithm === 'ppo' ? { entropyCoef: this.entropyCoef } : { discountFactor: this.discountFactor })
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

  private loadSimulatedModel(): void {
    // Simulate loading the model from localStorage
    const savedModel = localStorage.getItem('sudoku_ai_model');
    
    if (savedModel) {
      try {
        const modelData: ModelData = JSON.parse(savedModel);
        
        // Apply saved settings
        this.selectedAlgorithm = modelData.algorithm;
        this.learningRate = modelData.configuration.learningRate;
        this.batchSize = modelData.configuration.batchSize;
        
        if (modelData.algorithm === 'ppo' && modelData.configuration.entropyCoef) {
          this.entropyCoef = modelData.configuration.entropyCoef;
        } else if (modelData.algorithm === 'dqn' && modelData.configuration.discountFactor) {
          this.discountFactor = modelData.configuration.discountFactor;
        }
        
        // Apply saved stats
        this.accuracy = modelData.stats.accuracy;
        this.averageReward = modelData.stats.averageReward;
        this.episodes = modelData.stats.episodes;
        this.progress = 100; // Assume a loaded model is fully trained
        
        // Generate visualization based on loaded model
        this.generateNetworkVisualization();
        
        // Setup chart data for loaded model
        this.accuracyChartData = [{
          name: 'Accuracy', 
          series: [{ name: this.episodes.toString(), value: this.accuracy }]
        }];
        
        this.rewardChartData = [{
          name: 'Reward',
          series: [{ name: this.episodes.toString(), value: this.averageReward }]
        }];
        
        this.snackBar.open('Model loaded successfully!', 'Close', {
          duration: 3000
        });
      } catch (error) {
        console.error('Error parsing saved model:', error);
        this.snackBar.open('Failed to load model: Invalid model data', 'Close', {
          duration: 3000
        });
      }
    } else {
      this.snackBar.open('No saved model found!', 'Close', {
        duration: 3000
      });
    }
  }

  private updateChartsFromLoadedModel(): void {
    // Setup chart data for loaded model
    this.accuracyChartData = [{
      name: 'Accuracy', 
      series: [{ name: this.episodes.toString(), value: this.accuracy }]
    }];
    
    this.rewardChartData = [{
      name: 'Reward',
      series: [{ name: this.episodes.toString(), value: this.averageReward }]
    }];
  }

  private shuffleArray<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  // File handling methods for model import/export
  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const modelData = JSON.parse(e.target?.result as string);
          this.importModelFromFile(modelData);
        } catch (error) {
          console.error('Error parsing model file:', error);
          this.snackBar.open('Invalid model file format', 'Close', { duration: 3000 });
        }
      };
      reader.readAsText(file);
    }
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
      } else {
        this.snackBar.open('Unsupported model format', 'Close', { duration: 3000 });
      }
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
      } else {
        this.snackBar.open('No trained PPO model to export', 'Close', { duration: 3000 });
      }
    } catch (error) {
      console.error('Error exporting model:', error);
      this.snackBar.open('Failed to export model', 'Close', { duration: 3000 });
    }
  }

  triggerFileInput(): void {
    this.fileInput.nativeElement.click();
  }

  backToMenu(): void {
    // Clean up any training processes
    if (this.isTraining) {
      this.stopTraining();
    }
    
    // Dispose of PPO agent resources
    this.ppoAgent.dispose();
    
    this.router.navigate(['/menu']);
  }

  newBoard(): void {
    // Stop training if active
    if (this.isTraining) {
      this.stopTraining();
    }
    
    // Generate new board
    this.generateNewSudoku();
  }

  async onAlgorithmChange(): Promise<void> {
    // Stop any ongoing training
    if (this.isTraining) {
      this.stopTraining();
    }
    
    // Reset stats
    this.accuracy = 0;
    this.averageReward = 0;
    this.episodes = 0;
    this.progress = 0;
    this.actorLoss = 0;
    this.criticLoss = 0;
    this.avgAdvantage = 0;
    
    // Initialize the selected algorithm
    if (this.selectedAlgorithm === 'ppo') {
      await this.initializePPOAgent();
    }
    
    // Reset chart data
    this.accuracyChartData = [{ name: 'Accuracy', series: [] }];
    this.rewardChartData = [{ name: 'Reward', series: [] }];
    
    // Update network visualization
    this.generateNetworkVisualization();

    // Update neural network visualization state based on selected algorithm
    if (this.selectedAlgorithm === 'ppo') {
      this.hasNeuralNetwork = true;
      this.showNetworkVisualization = true;
      this.updateNetworkArchitecture();
    } else if (this.selectedAlgorithm === 'dqn') {
      this.hasNeuralNetwork = true;
      this.showNetworkVisualization = true;
      this.updateNetworkArchitecture();
    } else {
      this.hasNeuralNetwork = false;
      this.showNetworkVisualization = false;
    }
  }

  // Method to update the network architecture visualization based on the selected algorithm
  updateNetworkArchitecture() {
    if (this.selectedAlgorithm === 'ppo') {
      this.networkArchitecture = [
        { name: 'Input', units: 81 },  // Sudoku board state
        { name: 'Hidden 1', units: 128 },
        { name: 'Hidden 2', units: 64 },
        { name: 'Policy', units: 9 }   // Digit probabilities
      ];
    } else if (this.selectedAlgorithm === 'dqn') {
      this.networkArchitecture = [
        { name: 'Input', units: 81 },  // Sudoku board state
        { name: 'Hidden', units: 64 },
        { name: 'Value', units: 1 },   // State value
        { name: 'Advantage', units: 9 } // Action advantages
      ];
    } else {
      this.networkArchitecture = [];
    }
  }

  // Helper method to limit the number of nodes displayed in the visualization
  getVisibleNodes(totalUnits: number): number[] {
    // For large layers, just show a small representative sample
    const maxVisible = 5;
    if (totalUnits <= maxVisible) {
      return Array(totalUnits).fill(0).map((_, i) => i);
    } else {
      return Array(maxVisible).fill(0).map((_, i) => i);
    }
  }

  // Update PPO configuration when settings change
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

  // Helper method to handle cell clicks in the Sudoku board
  onCellClick(row: number, col: number): void {
    // For visualization purposes only - not interactive during training
    if (!this.isTraining) {
      console.log(`Cell clicked: (${row}, ${col})`);
    }
  }
}