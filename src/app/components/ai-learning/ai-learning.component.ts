import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSliderModule } from '@angular/material/slider';
import { FormsModule } from '@angular/forms';
import { CellComponent } from '../cell/cell.component';
import { SudokuService } from '../../services/sudoku.service';

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
    CellComponent
  ],
  templateUrl: './ai-learning.component.html',
  styleUrl: './ai-learning.component.css'
})
export class AiLearningComponent implements OnInit {
  // Add Math property to make it available in the template
  Math = Math;
  
  loading = true;
  board: number[][] = [];
  solution: number[][] = [];
  selectedAlgorithm: 'ppo' | 'dqn' = 'ppo';
  isTraining = false;
  progress = 0;
  trainingSpeed = 50; // Default speed (0-100)
  episodes = 0;
  currentReward = 0;
  averageReward = 0;
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
    private sudokuService: SudokuService
  ) {}

  ngOnInit(): void {
    // Initialize board
    this.initializeBoard();
  }

  private initializeBoard(): void {
    this.loading = true;
    this.sudokuService.getNewBoard().subscribe(boardData => {
      this.board = JSON.parse(JSON.stringify(boardData.grid));
      this.solution = JSON.parse(JSON.stringify(boardData.solution));
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

  startTraining(): void {
    if (this.isTraining) {
      this.stopTraining();
      return;
    }
    
    this.isTraining = true;
    this.progress = 0;
    this.episodes = 0;
    
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
      
      // Generate fake network visualization data
      this.generateNetworkVisualization();
      
      // Continue training until reaching 100%
      if (this.progress < 100) {
        const speed = 1000 - (this.trainingSpeed * 9); // Convert 0-100 to 1000-100ms
        setTimeout(trainingStep, speed);
      } else {
        // Training complete
        this.progress = 100;
        this.isTraining = false;
      }
    };
    
    // Start training loop
    trainingStep();
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