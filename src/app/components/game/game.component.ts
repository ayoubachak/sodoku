import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { SudokuService, SudokuBoard } from '../../services/sudoku.service';
import { CellComponent } from '../cell/cell.component';

@Component({
  selector: 'app-game',
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    MatButtonModule, 
    MatIconModule, 
    MatSnackBarModule,
    CellComponent
  ],
  templateUrl: './game.component.html',
  styleUrl: './game.component.css'
})
export class GameComponent implements OnInit {
  board: number[][] = [];
  solution: number[][] = [];
  originalBoard: number[][] = [];
  difficulty: string = 'medium';
  selectedCell: { row: number, col: number } | null = null;
  loading: boolean = true;
  timer: number = 0;
  timerInterval: any;
  mistakes: number = 0;
  gameCompleted: boolean = false;

  constructor(
    private sudokuService: SudokuService,
    private route: ActivatedRoute,
    private router: Router,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      if (params['difficulty']) {
        this.difficulty = params['difficulty'];
      }
      this.loadNewGame();
    });
  }

  loadNewGame(): void {
    this.loading = true;
    this.mistakes = 0;
    this.gameCompleted = false;
    this.resetTimer();
    
    this.sudokuService.getNewBoard(this.difficulty).subscribe({
      next: (data: SudokuBoard) => {
        this.board = JSON.parse(JSON.stringify(data.grid));
        this.solution = data.solution;
        this.originalBoard = JSON.parse(JSON.stringify(data.grid));
        this.difficulty = data.difficulty;
        this.loading = false;
        this.startTimer();
      },
      error: (error) => {
        console.error('Error loading Sudoku board:', error);
        this.snackBar.open('Failed to load Sudoku board. Please try again.', 'Close', {
          duration: 3000
        });
        this.loading = false;
      }
    });
  }

  selectCell(row: number, col: number): void {
    if (this.originalBoard[row][col] === 0) {
      this.selectedCell = { row, col };
    }
  }

  isSelected(row: number, col: number): boolean {
    return this.selectedCell?.row === row && this.selectedCell?.col === col;
  }

  isOriginal(row: number, col: number): boolean {
    return this.originalBoard[row][col] !== 0;
  }

  isSameNumber(row: number, col: number, num: number): boolean {
    return this.board[row][col] === num && num !== 0;
  }

  enterNumber(num: number): void {
    if (this.selectedCell && !this.gameCompleted) {
      const { row, col } = this.selectedCell;
      
      // Only allow changing non-original cells
      if (this.originalBoard[row][col] === 0) {
        // Check if the move is valid against the solution
        if (num === this.solution[row][col] || num === 0) {
          this.board[row][col] = num;
          
          // Check if the game is completed
          if (this.sudokuService.isSolved(this.board)) {
            this.gameCompleted = true;
            this.stopTimer();
            this.snackBar.open('Congratulations! You solved the puzzle!', 'Close', {
              duration: 5000
            });
          }
        } else {
          // Increment mistakes counter
          this.mistakes++;
          this.snackBar.open('Incorrect number!', 'Close', {
            duration: 1000
          });
          
          if (this.mistakes >= 3) {
            this.snackBar.open('Game over! Too many mistakes.', 'Close', {
              duration: 3000
            });
          }
        }
      }
    }
  }

  clearCell(): void {
    if (this.selectedCell && !this.isOriginal(this.selectedCell.row, this.selectedCell.col)) {
      this.board[this.selectedCell.row][this.selectedCell.col] = 0;
    }
  }

  startTimer(): void {
    this.timer = 0;
    this.timerInterval = setInterval(() => {
      this.timer++;
    }, 1000);
  }

  stopTimer(): void {
    clearInterval(this.timerInterval);
  }

  resetTimer(): void {
    this.stopTimer();
    this.timer = 0;
  }

  formatTime(): string {
    const minutes = Math.floor(this.timer / 60);
    const seconds = this.timer % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }

  backToMenu(): void {
    this.stopTimer();
    this.router.navigate(['/menu']);
  }

  newGame(): void {
    this.loadNewGame();
  }
}
