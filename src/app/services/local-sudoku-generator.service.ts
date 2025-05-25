import { Injectable } from '@angular/core';

export interface LocalSudokuBoard {
  grid: number[][];
  solution: number[][];
  difficulty: string;
}

@Injectable({
  providedIn: 'root'
})
export class LocalSudokuGeneratorService {

  constructor() { }

  // Generate a new Sudoku board with specified difficulty
  generateBoard(difficulty: 'easy' | 'medium' | 'hard' | 'expert' = 'medium'): LocalSudokuBoard {
    // First generate a complete solved board
    const solution = this.generateCompleteSolution();
    
    // Then remove numbers based on difficulty
    const grid = this.removeNumbers(solution, difficulty);
    
    return {
      grid: this.deepCopy(grid),
      solution: this.deepCopy(solution),
      difficulty: difficulty
    };
  }

  private generateCompleteSolution(): number[][] {
    // Start with a base valid solution and randomize it
    const board: number[][] = Array(9).fill(null).map(() => Array(9).fill(0));
    
    // Fill the board using backtracking
    this.solveSudoku(board);
    
    // Shuffle the board to make it more random
    this.shuffleBoard(board);
    
    return board;
  }

  private solveSudoku(board: number[][]): boolean {
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (board[row][col] === 0) {
          // Try numbers 1-9 in random order
          const numbers = this.shuffleArray([1, 2, 3, 4, 5, 6, 7, 8, 9]);
          
          for (const num of numbers) {
            if (this.isValidPlacement(board, row, col, num)) {
              board[row][col] = num;
              
              if (this.solveSudoku(board)) {
                return true;
              }
              
              board[row][col] = 0;
            }
          }
          return false;
        }
      }
    }
    return true;
  }

  private isValidPlacement(board: number[][], row: number, col: number, num: number): boolean {
    // Check row
    for (let x = 0; x < 9; x++) {
      if (board[row][x] === num) return false;
    }

    // Check column
    for (let y = 0; y < 9; y++) {
      if (board[y][col] === num) return false;
    }

    // Check 3x3 box
    const startRow = Math.floor(row / 3) * 3;
    const startCol = Math.floor(col / 3) * 3;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (board[startRow + i][startCol + j] === num) return false;
      }
    }

    return true;
  }

  private removeNumbers(solution: number[][], difficulty: string): number[][] {
    const board = this.deepCopy(solution);
    
    // Determine how many cells to remove based on difficulty
    const cellsToRemove = this.getCellsToRemove(difficulty);
    
    let removed = 0;
    const attempts = cellsToRemove * 3; // Prevent infinite loops
    
    for (let attempt = 0; attempt < attempts && removed < cellsToRemove; attempt++) {
      const row = Math.floor(Math.random() * 9);
      const col = Math.floor(Math.random() * 9);
      
      if (board[row][col] !== 0) {
        const backup = board[row][col];
        board[row][col] = 0;
        
        // Check if the puzzle still has a unique solution
        if (this.hasUniqueSolution(board)) {
          removed++;
        } else {
          // Restore the number if removing it creates multiple solutions
          board[row][col] = backup;
        }
      }
    }
    
    return board;
  }

  private getCellsToRemove(difficulty: string): number {
    switch (difficulty) {
      case 'easy': return 35;      // Remove 35-40 numbers
      case 'medium': return 45;    // Remove 45-50 numbers
      case 'hard': return 55;      // Remove 55-60 numbers
      case 'expert': return 64;    // Remove 64+ numbers
      default: return 45;
    }
  }

  private hasUniqueSolution(board: number[][]): boolean {
    const testBoard = this.deepCopy(board);
    let solutionCount = 0;
    
    const countSolutions = (board: number[][]): void => {
      if (solutionCount > 1) return; // Early exit if multiple solutions found
      
      for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
          if (board[row][col] === 0) {
            for (let num = 1; num <= 9; num++) {
              if (this.isValidPlacement(board, row, col, num)) {
                board[row][col] = num;
                countSolutions(board);
                board[row][col] = 0;
              }
            }
            return;
          }
        }
      }
      solutionCount++;
    };
    
    countSolutions(testBoard);
    return solutionCount === 1;
  }

  private shuffleBoard(board: number[][]): void {
    // Shuffle rows within each 3x3 block
    for (let block = 0; block < 3; block++) {
      const rows = [block * 3, block * 3 + 1, block * 3 + 2];
      this.shuffleArray(rows);
      
      const tempBoard = this.deepCopy(board);
      for (let i = 0; i < 3; i++) {
        for (let col = 0; col < 9; col++) {
          board[block * 3 + i][col] = tempBoard[rows[i]][col];
        }
      }
    }
    
    // Shuffle columns within each 3x3 block
    for (let block = 0; block < 3; block++) {
      const cols = [block * 3, block * 3 + 1, block * 3 + 2];
      this.shuffleArray(cols);
      
      const tempBoard = this.deepCopy(board);
      for (let row = 0; row < 9; row++) {
        for (let i = 0; i < 3; i++) {
          board[row][block * 3 + i] = tempBoard[row][cols[i]];
        }
      }
    }
  }

  private shuffleArray<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  private deepCopy(arr: number[][]): number[][] {
    return arr.map(row => [...row]);
  }
}