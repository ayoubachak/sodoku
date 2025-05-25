import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { CellComponent } from '../cell/cell.component';
import { SudokuService, SudokuBoard } from '../../services/sudoku.service';
import { LocalSudokuGeneratorService } from '../../services/local-sudoku-generator.service';
import { TechniqueCoachService, TechniqueDetection, SudokuTechnique } from '../../services/technique-coach.service';
import { TechniqueDialogComponent } from '../technique-dialog/technique-dialog.component';

@Component({
  selector: 'app-game',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatTooltipModule, MatSnackBarModule, MatDialogModule, CellComponent],
  templateUrl: './game.component.html',
  styleUrl: './game.component.css'
})
export class GameComponent implements OnInit, OnDestroy {
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

  // Properties for technique coaching
  highlightedCells: { row: number, col: number }[] = [];
  activeDetection: TechniqueDetection | null = null;
  coachMode: boolean = false;
  teachingStep: 'identify' | 'explain' | 'apply' | 'complete' = 'identify';
  pendingTechniques: TechniqueDetection[] = [];
  currentTeachingIndex: number = -1;
  notesRequired: boolean = false;
  singleCellHintRequested: boolean = false;
  targetCellForHint: { row: number, col: number } | null = null;
  
  // Skip functionality properties
  skippedTechniqueTypes: SudokuTechnique[] = [];
  skipToEnd: boolean = false;

  constructor(
    private readonly router: Router, 
    private readonly sudokuService: SudokuService,
    private readonly localGenerator: LocalSudokuGeneratorService,
    private _snackBar: MatSnackBar,
    private techniqueCoachService: TechniqueCoachService,
    private dialog: MatDialog
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

    this.startNewGame();
  }

  ngOnDestroy(): void {
    this.stopTimer();
  }

  initializeNotes(): void {
    this.notes = Array(9).fill(null).map(() => 
      Array(9).fill(null).map(() => [])
    );
  }

  startNewGame(): void {
    this.loading = true;
    this.resetGame();
    
    // Check if we should use API or local generator
    const settings = this.getSettings();
    
    if (settings.useApi) {
      // Use online API
      this.sudokuService.getNewBoard().subscribe({
        next: (board: SudokuBoard) => {
          this.initializeGame(board.grid, board.solution, board.difficulty);
          this.loading = false;
        },
        error: (error) => {
          console.error('Error loading puzzle from API:', error);
          // Fallback to local generator if API fails
          this.generateLocalPuzzle(settings.localDifficulty);
        }
      });
    } else {
      // Use local generator
      this.generateLocalPuzzle(settings.localDifficulty);
    }
  }

  private generateLocalPuzzle(difficulty: string): void {
    try {
      const localBoard = this.localGenerator.generateBoard(difficulty as 'easy' | 'medium' | 'hard' | 'expert');
      this.initializeGame(localBoard.grid, localBoard.solution, localBoard.difficulty);
      this.loading = false;
    } catch (error) {
      console.error('Error generating local puzzle:', error);
      // Use fallback if local generation fails
      this.initializeGame(this.getFallbackPuzzle().grid, this.getFallbackPuzzle().solution, 'Medium');
      this.loading = false;
    }
  }

  private getSettings() {
    const savedSettings = localStorage.getItem('sudokuSettings');
    return savedSettings ? JSON.parse(savedSettings) : {
      useApi: true,
      localDifficulty: 'medium'
    };
  }

  private getFallbackPuzzle() {
    return {
      grid: [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
      ],
      solution: [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
      ]
    };
  }

  loadNewGame(): void {
    this.startNewGame();
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
          console.log('Congratulations! You solved the puzzle!');
        }
      } else {
        // Increment mistakes counter
        this.mistakes++;
        console.log('Incorrect number!');
        this._snackBar.open('Incorrect number!', 'Close', {
          duration: 2000,
        });
        
        if (this.mistakes >= 3) {
          console.log('Game over! Too many mistakes.');
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
    
    console.log('Auto-filled notes for all empty cells!');
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

  // Toggle "Teach Me" coach mode
  toggleCoachMode(): void {
    this.coachMode = !this.coachMode;
    if (!this.coachMode) {
      // Clear any highlights when exiting coach mode
      this.highlightedCells = [];
      this.activeDetection = null;
      this.pendingTechniques = [];
      this.currentTeachingIndex = -1;
      this.teachingStep = 'identify';
    } else {
      // When entering coach mode, start the step by step teaching
      this.startTeachingMode();
    }
  }

  // Check if a cell is highlighted by the technique coach
  isCoachHighlighted(row: number, col: number): boolean {
    return this.highlightedCells.some(cell => cell.row === row && cell.col === col);
  }

  // Start the teaching mode process which guides users step by step
  async startTeachingMode(): Promise<void> {
    // Clear previous state
    this.highlightedCells = [];
    this.pendingTechniques = [];
    
    // Check if there are empty cells without notes
    let hasEmptyCellsWithoutNotes = false;
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (this.board[row][col] === 0 && this.notes[row][col].length === 0) {
          hasEmptyCellsWithoutNotes = true;
          break;
        }
      }
      if (hasEmptyCellsWithoutNotes) break;
    }

    if (hasEmptyCellsWithoutNotes) {
      // Alert the user that notes are needed for effective teaching
      this.notesRequired = true;
      this._snackBar.open(
        'To effectively learn techniques, notes are needed for empty cells. Would you like to auto-fill notes?',
        'Auto-fill Notes',
        { duration: 10000 }
      ).onAction().subscribe(() => {
        this.autoFillAllNotes();
        this.continueTeachingProcess();
      });
    } else {
      this.continueTeachingProcess();
    }
  }

  // Continue the teaching process after ensuring notes are available
  async continueTeachingProcess(): Promise<void> {
    // Get all applicable techniques
    this.pendingTechniques = await this.techniqueCoachService.detectTechniques(this.board, this.notes);

    if (this.pendingTechniques.length === 0) {
      this._snackBar.open(
        'No applicable solving techniques found. Try filling in more notes or using direct elimination.',
        'Close',
        { duration: 5000 }
      );
      return;
    }

    // Start with the first technique
    this.currentTeachingIndex = 0;
    this.teachTechnique();
  }

  // Teach the current technique
  async teachTechnique(): Promise<void> {
    if (this.currentTeachingIndex < 0 || this.currentTeachingIndex >= this.pendingTechniques.length) {
      return;
    }

    // Skip techniques that the user has chosen to skip
    let technique = this.pendingTechniques[this.currentTeachingIndex];
    
    // If we should skip to the end, don't show any more techniques
    if (this.skipToEnd) {
      // Show a message that we're skipping to the end
      this._snackBar.open('Skipping all remaining techniques.', 'Close', {
        duration: 3000
      });
      this.currentTeachingIndex = this.pendingTechniques.length;
      this.highlightedCells = [];
      this.activeDetection = null;
      return;
    }
    
    // Skip techniques of types the user has chosen to skip
    while (this.skippedTechniqueTypes.includes(technique.technique)) {
      this.currentTeachingIndex++;
      if (this.currentTeachingIndex >= this.pendingTechniques.length) {
        // No more techniques to show
        this._snackBar.open('No more applicable techniques to show.', 'Close', {
          duration: 3000
        });
        this.highlightedCells = [];
        this.activeDetection = null;
        return;
      }
      technique = this.pendingTechniques[this.currentTeachingIndex];
    }
    
    this.activeDetection = technique;
    this.teachingStep = 'identify';

    // Clear previous highlights
    this.highlightedCells = [];

    // Highlight the target cell
    if (technique.targetCell) {
      this.highlightedCells.push(technique.targetCell);
      this.selectedCell = technique.targetCell;
    }

    // Also highlight related cells if any
    if (technique.relatedCells) {
      this.highlightedCells = [...this.highlightedCells, ...technique.relatedCells];
    }

    // Show a dialog explaining the technique
    const dialogRef = this.dialog.open(TechniqueDialogComponent, {
      data: {
        title: `Learn: ${technique.technique}`,
        message: `<p>${technique.detailExplanation || technique.explanation}</p>
                 <p>Do you want to apply this technique to solve this cell?</p>`,
        technique: technique.technique,
        position: technique.targetCell,
        value: technique.value,
        showSkipOptions: true,
        isTeachingMode: true,
        techniqueCount: this.pendingTechniques.length,
        currentIndex: this.currentTeachingIndex
      },
      width: '400px',
      panelClass: 'technique-dialog',
      hasBackdrop: false,
      autoFocus: false
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result === 'apply') {
        // Apply technique and move to the next one
        this.applyTechniqueAndContinue();
      } else if (result === 'skip-this') {
        // Skip just this instance and move to next technique
        this.skipCurrentTechnique();
      } else if (result === 'skip-apply') {
        // Apply the current technique and skip to the next one
        this.applyAndSkipToNext();
      } else if (result === 'skip-type') {
        // Skip all techniques of this type
        this.skipTechniqueType(technique.technique);
      } else if (result === 'skip-all') {
        // Skip to the end
        this.skipToEnd = true;
        this.highlightedCells = [];
        this.activeDetection = null;
      } else {
        // If user cancels, remain in coach mode but don't proceed
        this.highlightedCells = [];
        this.activeDetection = null;
      }
    });
  }

  // Apply the current technique and immediately skip to the next one
  applyAndSkipToNext(): void {
    if (!this.activeDetection || !this.activeDetection.targetCell || !this.activeDetection.value) {
      return;
    }

    const { row, col } = this.activeDetection.targetCell;
    const value = this.activeDetection.value;

    // Apply the technique (set the value in the board)
    this.board[row][col] = value;
    this.notes[row][col] = [];

    // Record technique usage
    this.techniqueCoachService.recordTechniqueUsage(this.activeDetection.technique, true);

    // Eliminate this number from related cells' notes
    this.eliminateNotesFromRelatedCells(row, col, value);

    // Check if the game is completed
    if (this.sudokuService.isSolved(this.board)) {
      this.gameCompleted = true;
      this.stopTimer();
      this.coachMode = false;
      return;
    }

    // Clear highlights and proceed to the next technique without delay
    this.highlightedCells = [];
    this.activeDetection = null;
    this.currentTeachingIndex++;

    if (this.currentTeachingIndex < this.pendingTechniques.length) {
      // Show the next technique immediately
      setTimeout(() => this.teachTechnique(), 100);
    } else {
      // No more techniques to teach at the moment
      this._snackBar.open(
        'You\'ve applied all available techniques! Try using Auto Notes again to find more.',
        'Continue',
        { duration: 6000 }
      ).onAction().subscribe(() => {
        this.autoFillAllNotes();
        this.startTeachingMode();
      });
    }
  }

  // Skip the current technique and move to the next one
  skipCurrentTechnique(): void {
    this.highlightedCells = [];
    this.activeDetection = null;
    this.currentTeachingIndex++;
    
    if (this.currentTeachingIndex < this.pendingTechniques.length) {
      // Show the next technique
      setTimeout(() => this.teachTechnique(), 100);
    } else {
      // No more techniques to teach
      this._snackBar.open(
        'You\'ve reviewed all available techniques!',
        'OK',
        { duration: 3000 }
      );
    }
  }

  // Skip all techniques of the specified type
  skipTechniqueType(techniqueType: SudokuTechnique): void {
    // Add this technique type to the list of skipped types
    if (!this.skippedTechniqueTypes.includes(techniqueType)) {
      this.skippedTechniqueTypes.push(techniqueType);
    }
    
    this.highlightedCells = [];
    this.activeDetection = null;
    
    // Find the next technique that isn't of the skipped type
    let foundNext = false;
    while (this.currentTeachingIndex < this.pendingTechniques.length) {
      this.currentTeachingIndex++;
      if (this.currentTeachingIndex >= this.pendingTechniques.length) {
        break;
      }
      
      const nextTechnique = this.pendingTechniques[this.currentTeachingIndex];
      if (!this.skippedTechniqueTypes.includes(nextTechnique.technique)) {
        foundNext = true;
        break;
      }
    }
    
    if (foundNext && this.currentTeachingIndex < this.pendingTechniques.length) {
      // Show the next non-skipped technique
      setTimeout(() => this.teachTechnique(), 100);
    } else {
      // No more non-skipped techniques to teach
      this._snackBar.open(
        'No more applicable techniques to show after skipping.',
        'OK',
        { duration: 3000 }
      );
    }
  }

  // Apply the current technique and continue to the next one
  applyTechniqueAndContinue(): void {
    if (!this.activeDetection || !this.activeDetection.targetCell || !this.activeDetection.value) {
      return;
    }

    const { row, col } = this.activeDetection.targetCell;
    const value = this.activeDetection.value;

    // Apply the technique (set the value in the board)
    this.board[row][col] = value;
    this.notes[row][col] = [];

    // Record technique usage
    this.techniqueCoachService.recordTechniqueUsage(this.activeDetection.technique, true);

    // Eliminate this number from related cells' notes
    this.eliminateNotesFromRelatedCells(row, col, value);

    // Clear highlights
    this.highlightedCells = [];
    this.activeDetection = null;

    // Check if the game is completed
    if (this.sudokuService.isSolved(this.board)) {
      this.gameCompleted = true;
      this.stopTimer();
      this.coachMode = false;
      return;
    }

    // Move to the next technique
    this.currentTeachingIndex++;
    if (this.currentTeachingIndex < this.pendingTechniques.length) {
      // Slight delay before showing the next technique
      setTimeout(() => this.teachTechnique(), 500);
    } else {
      // No more techniques to teach at the moment
      this._snackBar.open(
        'You\'ve applied all available techniques! Try using Auto Notes again to find more.',
        'Continue',
        { duration: 6000 }
      ).onAction().subscribe(() => {
        this.autoFillAllNotes();
        this.startTeachingMode();
      });
    }
  }

  // Get an AI-powered hint that explains the technique
  async getSmartHint(): Promise<void> {
    // If a specific cell is selected and we're not in coach mode, try to get a hint for that cell
    if (this.selectedCell && !this.coachMode) {
      await this.getHintForSelectedCell();
      return;
    }

    // Standard hint behavior for coach mode or when no cell is selected
    this.singleCellHintRequested = false;
    this.targetCellForHint = null;
    
    // Clear previous highlights
    this.highlightedCells = [];
    
    // Only show a limited number of hints
    if (this.hintsUsed >= this.maxHints) {
      this._snackBar.open('No more hints available!', 'Close', {
        duration: 3000,
      });
      return;
    }

    // Get a technique-based hint from the coach service
    const hint = await this.techniqueCoachService.getHint(this.board, this.notes);
    
    if (!hint) {
      this._snackBar.open('No applicable techniques found.', 'Close', {
        duration: 2000
      });
      return;
    }
    
    // Store the active detection for UI rendering
    this.activeDetection = hint;
    
    // Add the target cell to highlighted cells
    if (hint.targetCell) {
      this.highlightedCells.push(hint.targetCell);
      
      // Select the target cell
      this.selectedCell = hint.targetCell;
    }
    
    // Add related cells to highlighted cells
    if (hint.relatedCells) {
      this.highlightedCells = [...this.highlightedCells, ...hint.relatedCells];
    }
    
    // If we're in coach mode, show a more detailed dialog
    if (this.coachMode) {
      const dialogRef = this.dialog.open(TechniqueDialogComponent, {
        data: {
          title: `Hint: ${hint.technique}`,
          message: `<p>${hint.detailExplanation || hint.explanation}</p>`,
          technique: hint.technique,
          position: hint.targetCell,
          value: hint.value
        },
        width: '400px'
      });

      dialogRef.afterClosed().subscribe(result => {
        if (result === 'apply') {
          this.applyHint(hint);
        } else {
          // If user cancels, clear highlights
          this.highlightedCells = [];
          this.activeDetection = null;
        }
      });
    } else {
      // Otherwise, show the simple snackbar
      this._snackBar.open(
        `${hint.technique}: ${hint.explanation}`, 
        'Apply', 
        {
          duration: 5000,
          panelClass: 'technique-hint-snackbar'
        }
      ).onAction().subscribe(() => {
        this.applyHint(hint);
      });
    }
  }

  // Get a hint for the currently selected cell
  async getHintForSelectedCell(): Promise<void> {
    if (!this.selectedCell) return;
    
    this.singleCellHintRequested = true;
    this.targetCellForHint = this.selectedCell;
    const { row, col } = this.selectedCell;

    // Only show a limited number of hints
    if (this.hintsUsed >= this.maxHints) {
      this._snackBar.open('No more hints available!', 'Close', {
        duration: 3000,
      });
      return;
    }

    // If the cell is already filled
    if (this.board[row][col] !== 0) {
      this._snackBar.open('This cell is already filled.', 'Close', {
        duration: 2000
      });
      return;
    }

    // Try to get a hint specifically for this cell
    const hint = await this.techniqueCoachService.getHintForCell(this.board, this.notes, row, col);
    
    if (hint) {
      // We found a technique for this cell
      this.activeDetection = hint;
      
      // Highlight the related cells if any
      this.highlightedCells = [];
      if (hint.relatedCells) {
        this.highlightedCells = [...hint.relatedCells];
      }
      
      // Open a dialog to explain the technique
      const dialogRef = this.dialog.open(TechniqueDialogComponent, {
        data: {
          title: `Hint for Selected Cell: ${hint.technique}`,
          message: `<p>${hint.detailExplanation || hint.explanation}</p>`,
          technique: hint.technique,
          position: { row, col },
          value: hint.value,
          showRevealOption: true
        },
        width: '400px'
      });

      dialogRef.afterClosed().subscribe(result => {
        if (result === 'apply') {
          // Apply the hint (technique)
          this.applyHint(hint);
        } else if (result === 'reveal') {
          // Just reveal the value
          this.revealCellValue(row, col);
        } else {
          // If user cancels, clear highlights
          this.highlightedCells = [];
          this.activeDetection = null;
        }
      });
    } else {
      // No technique found for this cell, offer to reveal the value
      const dialogRef = this.dialog.open(TechniqueDialogComponent, {
        data: {
          title: 'No Technique Available',
          message: `<p>There are no applicable techniques for this cell at the moment. Would you like to reveal the value?</p>`,
          position: { row, col },
          showRevealOption: true
        },
        width: '400px'
      });

      dialogRef.afterClosed().subscribe(result => {
        if (result === 'reveal') {
          this.revealCellValue(row, col);
        }
      });
    }
  }

  // Directly reveal a cell's value from the solution
  revealCellValue(row: number, col: number): void {
    if (this.board[row][col] !== 0) return;
    
    const value = this.solution[row][col];
    this.board[row][col] = value;
    this.notes[row][col] = [];
    this.hintsUsed++;
    
    // Eliminate this number from related cells' notes
    this.eliminateNotesFromRelatedCells(row, col, value);
    
    // Check if the game is completed
    if (this.sudokuService.isSolved(this.board)) {
      this.gameCompleted = true;
      this.stopTimer();
    }
  }

  // Apply a hint to the board
  applyHint(hint: TechniqueDetection): void {
    if (hint.targetCell && hint.value) {
      const { row, col } = hint.targetCell;
      this.board[row][col] = hint.value;
      this.notes[row][col] = [];
      this.hintsUsed++;
      
      // Record that this technique was used
      this.techniqueCoachService.recordTechniqueUsage(hint.technique, true);
      
      // Eliminate this number from related cells' notes
      this.eliminateNotesFromRelatedCells(row, col, hint.value);
      
      // Clear highlights after applying
      this.highlightedCells = [];
      this.activeDetection = null;
      
      // Check if the game is completed
      if (this.sudokuService.isSolved(this.board)) {
        this.gameCompleted = true;
        this.stopTimer();
      }
    }
  }

  // Legacy hint method (now calls smart hint)
  getHint(): void {
    this.getSmartHint();
  }

  backToMenu(): void {
    this.stopTimer();
    this.router.navigate(['/menu']);
  }

  newGame(): void {
    this.startNewGame();
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

  // Format time (MM:SS)
  formatTime(): string {
    const minutes = Math.floor(this.timer / 60);
    const seconds = this.timer % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }
  
  // Start timer
  startTimer(): void {
    this.stopTimer(); // Clear any existing timer
    this.timer = 0;
    this.timerInterval = setInterval(() => {
      this.timer++;
    }, 1000);
  }
  
  // Stop timer
  stopTimer(): void {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  }

  private initializeGame(grid: number[][], solution: number[][], difficulty: string): void {
    this.board = JSON.parse(JSON.stringify(grid));
    this.solution = JSON.parse(JSON.stringify(solution));
    this.originalBoard = JSON.parse(JSON.stringify(grid));
    this.difficulty = difficulty;
    this.loading = false;
    this.startTimer();
  }

  private resetGame(): void {
    this.board = Array(9).fill(null).map(() => Array(9).fill(0));
    this.notes = Array(9).fill(null).map(() => Array(9).fill(null).map(() => []));
    this.selectedCell = null;
    this.loading = true;
    this.timer = 0;
    this.mistakes = 0;
    this.gameCompleted = false;
    this.notesMode = false;
    this.hintsUsed = 0;
  }
}
