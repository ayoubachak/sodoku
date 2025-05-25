import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { SudokuService, SudokuBoard } from '../../services/sudoku.service';
import { ThemeService } from '../../services/theme.service';
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
    MatTooltipModule,
    CellComponent
  ],
  templateUrl: './game.component.html',
  styleUrl: './game.component.css'
})
export class GameComponent implements OnInit {
  board: number[][] = [];
  notes: number[][][] = []; // 9x9 grid of arrays for notes
  solution: number[][] = [];
  originalBoard: number[][] = [];
  difficulty: string = 'medium';
  selectedCell: { row: number, col: number } | null = null;
  loading: boolean = true;
  timer: number = 0;
  timerInterval: any;
  mistakes: number = 0;
  gameCompleted: boolean = false;
  notesMode: boolean = false;
  hintsUsed: number = 0;
  maxHints: number = 3;

  // Make Math available in template
  Math = Math;

  constructor(
    private readonly sudokuService: SudokuService,
    private readonly route: ActivatedRoute,
    private readonly router: Router,
    private readonly snackBar: MatSnackBar
  ) {
    // Initialize notes array
    this.initializeNotes();
  }

  ngOnInit(): void {
    // Load theme preference and apply it
    const savedTheme = localStorage.getItem('sudoku-theme');
    if (savedTheme === 'dark') {
      document.body.classList.add('dark-theme');
      document.body.classList.remove('light-theme');
    } else {
      document.body.classList.add('light-theme');
      document.body.classList.remove('dark-theme');
    }

    this.route.queryParams.subscribe(params => {
      if (params['difficulty']) {
        this.difficulty = params['difficulty'];
      }
      this.loadNewGame();
    });
  }

  initializeNotes(): void {
    this.notes = Array(9).fill(null).map(() => 
      Array(9).fill(null).map(() => [])
    );
  }

  loadNewGame(): void {
    this.loading = true;
    this.mistakes = 0;
    this.gameCompleted = false;
    this.hintsUsed = 0;
    this.notesMode = false;
    this.resetTimer();
    this.initializeNotes();
    
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
    this.selectedCell = { row, col };
  }

  isSelected(row: number, col: number): boolean {
    return this.selectedCell?.row === row && this.selectedCell?.col === col;
  }

  isOriginal(row: number, col: number): boolean {
    return this.originalBoard[row][col] !== 0;
  }

  isSameNumber(row: number, col: number): boolean {
    if (!this.selectedCell || this.board[row][col] === 0) return false;
    const selectedValue = this.board[this.selectedCell.row][this.selectedCell.col];
    return this.board[row][col] === selectedValue && selectedValue !== 0;
  }

  isHighlighted(row: number, col: number): boolean {
    if (!this.selectedCell) return false;
    return this.selectedCell.row === row || this.selectedCell.col === col ||
           this.getSameBox(this.selectedCell.row, this.selectedCell.col, row, col);
  }

  getSameBox(row1: number, col1: number, row2: number, col2: number): boolean {
    const box1 = Math.floor(row1 / 3) * 3 + Math.floor(col1 / 3);
    const box2 = Math.floor(row2 / 3) * 3 + Math.floor(col2 / 3);
    return box1 === box2;
  }

  enterNumber(num: number): void {
    if (!this.selectedCell || this.gameCompleted) return;
    
    const { row, col } = this.selectedCell;
    
    // Only allow changing non-original cells
    if (this.originalBoard[row][col] !== 0) return;

    if (this.notesMode) {
      this.toggleNote(row, col, num);
    } else {
      // Clear notes when entering a number
      this.notes[row][col] = [];
      
      // Check if the move is valid against the solution
      if (num === this.solution[row][col]) {
        this.board[row][col] = num;
        
        // Automatically eliminate this number from related cells' notes
        this.eliminateNotesFromRelatedCells(row, col, num);
        
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

  eliminateNotesFromRelatedCells(row: number, col: number, num: number): void {
    // Eliminate from same row
    for (let c = 0; c < 9; c++) {
      if (c !== col && this.board[row][c] === 0) {
        this.removeNoteFromCell(row, c, num);
      }
    }
    
    // Eliminate from same column
    for (let r = 0; r < 9; r++) {
      if (r !== row && this.board[r][col] === 0) {
        this.removeNoteFromCell(r, col, num);
      }
    }
    
    // Eliminate from same 3x3 box
    const boxStartRow = Math.floor(row / 3) * 3;
    const boxStartCol = Math.floor(col / 3) * 3;
    
    for (let r = boxStartRow; r < boxStartRow + 3; r++) {
      for (let c = boxStartCol; c < boxStartCol + 3; c++) {
        if ((r !== row || c !== col) && this.board[r][c] === 0) {
          this.removeNoteFromCell(r, c, num);
        }
      }
    }
  }

  removeNoteFromCell(row: number, col: number, num: number): void {
    const noteIndex = this.notes[row][col].indexOf(num);
    if (noteIndex > -1) {
      this.notes[row][col].splice(noteIndex, 1);
    }
  }

  // Auto-fill notes for a cell based on current board state
  autoFillNotes(row: number, col: number): void {
    if (this.board[row][col] !== 0) return; // Cell already filled
    
    const possibleNumbers: number[] = [];
    
    for (let num = 1; num <= 9; num++) {
      if (this.isValidPlacement(row, col, num)) {
        possibleNumbers.push(num);
      }
    }
    
    this.notes[row][col] = possibleNumbers;
  }

  isValidPlacement(row: number, col: number, num: number): boolean {
    // Check row
    for (let c = 0; c < 9; c++) {
      if (this.board[row][c] === num) return false;
    }
    
    // Check column
    for (let r = 0; r < 9; r++) {
      if (this.board[r][col] === num) return false;
    }
    
    // Check 3x3 box
    const boxStartRow = Math.floor(row / 3) * 3;
    const boxStartCol = Math.floor(col / 3) * 3;
    
    for (let r = boxStartRow; r < boxStartRow + 3; r++) {
      for (let c = boxStartCol; c < boxStartCol + 3; c++) {
        if (this.board[r][c] === num) return false;
      }
    }
    
    return true;
  }

  // Auto-fill all empty cells with possible notes
  autoFillAllNotes(): void {
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (this.board[row][col] === 0) {
          this.autoFillNotes(row, col);
        }
      }
    }
    
    this.snackBar.open('Auto-filled notes for all empty cells!', 'Close', {
      duration: 2000
    });
  }

  toggleNote(row: number, col: number, num: number): void {
    if (this.board[row][col] !== 0) return; // Can't add notes to filled cells
    
    const noteIndex = this.notes[row][col].indexOf(num);
    if (noteIndex > -1) {
      this.notes[row][col].splice(noteIndex, 1);
    } else {
      this.notes[row][col].push(num);
      this.notes[row][col].sort((a, b) => a - b);
    }
  }

  clearCell(): void {
    if (this.selectedCell && !this.isOriginal(this.selectedCell.row, this.selectedCell.col)) {
      this.board[this.selectedCell.row][this.selectedCell.col] = 0;
      this.notes[this.selectedCell.row][this.selectedCell.col] = [];
    }
  }

  toggleNotesMode(): void {
    this.notesMode = !this.notesMode;
  }

  getHint(): void {
    if (this.hintsUsed >= this.maxHints || !this.selectedCell) {
      this.snackBar.open('No more hints available!', 'Close', { duration: 2000 });
      return;
    }

    const { row, col } = this.selectedCell;
    if (this.originalBoard[row][col] !== 0) {
      this.snackBar.open('Cell already filled!', 'Close', { duration: 2000 });
      return;
    }

    this.board[row][col] = this.solution[row][col];
    this.notes[row][col] = [];
    this.hintsUsed++;
    
    this.snackBar.open(`Hint used! ${this.maxHints - this.hintsUsed} remaining`, 'Close', {
      duration: 2000
    });

    if (this.sudokuService.isSolved(this.board)) {
      this.gameCompleted = true;
      this.stopTimer();
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

  // Get board organized into 3x3 sections for proper display
  getSections(): any[] {
    const sections = [];
    
    for (let sectionRow = 0; sectionRow < 3; sectionRow++) {
      for (let sectionCol = 0; sectionCol < 3; sectionCol++) {
        const section = [];
        
        for (let cellRow = 0; cellRow < 3; cellRow++) {
          for (let cellCol = 0; cellCol < 3; cellCol++) {
            const row = sectionRow * 3 + cellRow;
            const col = sectionCol * 3 + cellCol;
            
            section.push({
              value: this.board[row][col],
              notes: this.notes[row][col],
              isOriginal: this.isOriginal(row, col),
              isSelected: this.isSelected(row, col),
              isSameNumber: this.isSameNumber(row, col),
              isHighlighted: this.isHighlighted(row, col),
              row: row,
              col: col
            });
          }
        }
        
        sections.push(section);
      }
    }
    
    return sections;
  }
}
