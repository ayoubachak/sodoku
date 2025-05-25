import { Component, OnInit, OnDestroy, AfterViewInit, ChangeDetectorRef } from '@angular/core';
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
  learningRate = 0.001;
  batchSize = 64;
  entropyCoef = 0.05;
  discountFactor = 0.99;
  
  // Training stats
  currentReward = 0;
  averageReward = 0;
  accuracy = 0;
  
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

  constructor(
    private router: Router,
    private sudokuService: SudokuService,
    private snackBar: MatSnackBar,
    private cdr: ChangeDetectorRef
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
      this.thinkingQValues.push(Math.random() * 5);
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

  ngOnInit(): void {
    // Initialize board
    this.initializeBoard();
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
  }

  private initializeBoard(): void {
    this.loading = true;
    this.sudokuService.getNewBoard().subscribe(boardData => {
      this.board = JSON.parse(JSON.stringify(boardData.grid));
      this.solution = JSON.parse(JSON.stringify(boardData.solution));
      this.initializeCellData();
      this.loading = false;
      this.showModelPerformance = false;
      this.cellHighlights = [];
    });
  }

  private initializeCellData(): void {
    this.cellData = [];
    for (let row = 0; row < 9; row++) {
      this.cellData[row] = [];
      for (let col = 0; col < 9; col++) {
        this.cellData[row][col] = {
          row,
          col,
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
        for (let rowOffset = 0; rowOffset < 3; rowOffset++) {
          for (let colOffset = 0; colOffset < 3; colOffset++) {
            const row = sectionRow * 3 + rowOffset;
            const col = sectionCol * 3 + colOffset;
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

  startTraining(): void {
    if (this.isTraining) {
      this.stopTraining();
      return;
    }
    
    this.isTraining = true;
    this.progress = 0;
    this.episodes = 0;
    this.accuracy = 0;
    this.averageReward = 0;
    
    // Reset chart data
    this.accuracyChartData = [{ name: 'Accuracy', series: [] }];
    this.rewardChartData = [{ name: 'Reward', series: [] }];
    
    // Simulate training with intervals
    this.trainWithAlgorithm();
  }

  stopTraining(): void {
    this.isTraining = false;
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

  trainWithAlgorithm(): void {
    // This is a simulation of training
    // In a real implementation, you would use TensorFlow.js here
    
    const trainingStep = () => {
      if (!this.isTraining) return;
      
      // Update progress
      this.progress += 0.5;
      this.episodes += 1;
      
      // Simulate rewards
      this.currentReward = Math.random() * 2 - 0.5; // -0.5 to 1.5
      this.averageReward = (this.averageReward * (this.episodes - 1) + this.currentReward) / this.episodes;
      
      // Simulate accuracy (increasing over time)
      this.accuracy = Math.min(95, (this.progress / 100) * 90 + Math.random() * 10);
      
      // Simulate AI focusing on specific cells
      this.resetHighlights();
      const numHighlights = Math.floor(Math.random() * 3) + 1;
      for (let i = 0; i < numHighlights; i++) {
        const row = Math.floor(Math.random() * 9);
        const col = Math.floor(Math.random() * 9);
        if (this.board[row][col] === 0) { // Only highlight empty cells
          this.cellData[row][col].isHighlighted = true;
          
          // Add to highlights array for visualization
          this.cellHighlights.push({
            row,
            col,
            value: Math.floor(Math.random() * 9) + 1
          });
        }
      }
      
      // Keep only the last 5 highlights
      if (this.cellHighlights.length > 5) {
        this.cellHighlights = this.cellHighlights.slice(-5);
      }
      
      // Generate fake network visualization data
      this.generateNetworkVisualization();
      
      // Update chart data
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
      ? [81, 32, 16, 729] // 81 input cells, 729 output actions (81 cells * 9 digits)
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

  testModel(): void {
    // Simulate testing the model on the current board
    this.resetHighlights();
    
    // Highlight a sequence of moves to simulate the model solving the puzzle
    let emptyCells = [];
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (this.board[row][col] === 0) {
          emptyCells.push({ row, col });
        }
      }
    }
    
    // Shuffle the empty cells to simulate non-linear solving
    emptyCells = this.shuffleArray(emptyCells);
    
    let cellIndex = 0;
    const solveStep = () => {
      if (cellIndex >= emptyCells.length) {
        // All cells filled
        return;
      }
      
      this.resetHighlights();
      const { row, col } = emptyCells[cellIndex];
      this.cellData[row][col].isHighlighted = true;
      
      // Fill with the correct value from solution
      setTimeout(() => {
        this.board[row][col] = this.solution[row][col];
        this.cellData[row][col].value = this.solution[row][col];
        cellIndex++;
        
        if (cellIndex < emptyCells.length) {
          setTimeout(solveStep, 200);
        }
      }, 300);
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
          emptyCells.push({ row, col });
          this.totalTestCells++;
        }
      }
    }
    
    // Simulate the actual testing
    const startTime = performance.now();
    
    // Shuffle the empty cells
    emptyCells = this.shuffleArray(emptyCells);
    
    // Process cells with a delay to visualize the process
    let cellIndex = 0;
    const processCell = () => {
      if (cellIndex >= emptyCells.length) {
        // Testing complete
        const endTime = performance.now();
        this.testTime = Math.round(endTime - startTime);
        this.testAccuracy = (this.correctCells / this.totalTestCells) * 100;
        return;
      }
      
      this.resetHighlights();
      const { row, col } = emptyCells[cellIndex];
      this.cellData[row][col].isHighlighted = true;
      
      // Determine if the model gets this cell correct (simulate with probability based on training progress)
      // Higher training progress = higher chance of being correct
      const correctProb = 0.5 + (this.progress / 200); // 50% to 100% chance
      const isCorrect = Math.random() < correctProb;
      
      setTimeout(() => {
        if (isCorrect) {
          this.board[row][col] = this.solution[row][col];
          this.cellData[row][col].value = this.solution[row][col];
          this.correctCells++;
        } else {
          // Fill with a wrong value
          let wrongValue;
          do {
            wrongValue = Math.floor(Math.random() * 9) + 1;
          } while (wrongValue === this.solution[row][col]);
          
          this.board[row][col] = wrongValue;
          this.cellData[row][col].value = wrongValue;
        }
        
        cellIndex++;
        if (cellIndex < emptyCells.length) {
          setTimeout(processCell, 100);
        } else {
          // Show final stats
          const endTime = performance.now();
          this.testTime = Math.round(endTime - startTime);
          this.testAccuracy = (this.correctCells / this.totalTestCells) * 100;
        }
      }, 200);
    };
    
    processCell();
  }
  
  saveModel(): void {
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
  
  loadModel(): void {
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

  shuffleArray<T>(array: T[]): T[] {
    const newArray = [...array];
    for (let i = newArray.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
    }
    return newArray;
  }

  newBoard(): void {
    this.initializeBoard();
  }

  backToMenu(): void {
    this.router.navigate(['/menu']);
  }
  
  // Helper method to calculate additional nodes not shown in the visualization
  getMoreNodesCount(layer: any[]): number {
    // Calculate how many nodes are not being displayed
    return layer.length - layer.filter(n => n).length;
  }
}