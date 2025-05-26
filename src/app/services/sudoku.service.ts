import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { LocalSudokuGeneratorService } from './local-sudoku-generator.service';

export interface SudokuResponse {
  newboard: {
    grids: {
      value: number[][];
      solution: number[][];
      difficulty: string;
    }[];
    results: number;
    message: string;
  };
}

export interface SudokuBoard {
  grid: number[][];
  solution: number[][];
  difficulty: string;
  notes?: number[][][]; // 9x9 grid of arrays containing possible numbers
}

export interface CellData {
  value: number;
  notes: number[];
  isOriginal: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class SudokuService {
  private apiUrl = 'https://sudoku-api.vercel.app/api/dosuku';

  constructor(
    private http: HttpClient,
    private localGenerator: LocalSudokuGeneratorService
  ) { }

  // Get a new Sudoku board with specified difficulty, respecting user settings or AI settings
  getNewBoard(difficulty?: string, useAiSettings?: boolean): Observable<SudokuBoard> {
    let settings;
    
    if (useAiSettings) {
      // For AI Learning component, check for AI-specific settings
      settings = this.getAiSettings();
    } else {
      // For regular game, use normal settings
      settings = this.getSettings();
    }
    
    if (settings.useApi) {
      return this.getApiBoard(difficulty, useAiSettings);
    } else {
      const localDifficulty = useAiSettings ? settings.aiDifficulty : settings.localDifficulty;
      return this.getLocalBoard(localDifficulty as 'easy' | 'medium' | 'hard' | 'expert');
    }
  }

  // Get board from API
  private getApiBoard(difficulty?: string, useAiSettings?: boolean): Observable<SudokuBoard> {
    // GraphQL query to get a complete board with value, solution, and difficulty
    const query = '{newboard(limit:1){grids{value,solution,difficulty},results,message}}';
    
    return this.http.get<SudokuResponse>(`${this.apiUrl}?query=${encodeURIComponent(query)}`)
      .pipe(
        map(response => {
          const grid = response.newboard.grids[0];
          
          return {
            grid: grid.value,
            solution: grid.solution,
            difficulty: grid.difficulty
          };
        }),
        catchError(error => {
          console.error('Error fetching Sudoku board from API:', error);
          console.log('Falling back to local generation...');
          // Fallback to local generation if API fails
          const settings = useAiSettings ? this.getAiSettings() : this.getSettings();
          const fallbackDifficulty = useAiSettings ? settings.aiDifficulty : settings.localDifficulty;
          return this.getLocalBoard(fallbackDifficulty as 'easy' | 'medium' | 'hard' | 'expert');
        })
      );
  }

  // Get board from local generator
  private getLocalBoard(difficulty: 'easy' | 'medium' | 'hard' | 'expert'): Observable<SudokuBoard> {
    try {
      const localBoard = this.localGenerator.generateBoard(difficulty);
      
      return of({
        grid: localBoard.grid,
        solution: localBoard.solution,
        difficulty: localBoard.difficulty
      });
    } catch (error) {
      console.error('Error generating local Sudoku board:', error);
      // Return fallback board if local generation fails
      return of(this.getFallbackBoard());
    }
  }

  // Fallback board in case API fails
  private getFallbackBoard(): SudokuBoard {
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
      ],
      difficulty: 'Medium'
    };
  }

  // Get AI Learning specific settings
  private getAiSettings(): any {
    const savedSettings = localStorage.getItem('sudokuAiSettings');
    if (savedSettings) {
      return JSON.parse(savedSettings);
    }
    
    // Default AI settings if none saved
    return {
      useApi: true,
      aiDifficulty: 'medium'
    };
  }

  // Get user settings from localStorage
  private getSettings(): any {
    const savedSettings = localStorage.getItem('sudokuSettings');
    if (savedSettings) {
      return JSON.parse(savedSettings);
    }
    
    // Default settings if none saved
    return {
      useApi: true,
      localDifficulty: 'medium',
      showMistakes: true,
      showTimer: true,
      highlightSameNumbers: true,
      theme: 'light'
    };
  }

  // Save AI Learning specific settings
  saveAiSettings(settings: any): void {
    localStorage.setItem('sudokuAiSettings', JSON.stringify(settings));
  }

  // Get board for AI Learning specifically
  getAiBoard(): Observable<SudokuBoard> {
    return this.getNewBoard(undefined, true);
  }

  // Force API usage (for testing or specific needs)
  forceApiBoard(difficulty?: string): Observable<SudokuBoard> {
    return this.getApiBoard(difficulty, false);
  }

  // Force local generation (for testing or specific needs)
  forceLocalBoard(difficulty: 'easy' | 'medium' | 'hard' | 'expert' = 'medium'): Observable<SudokuBoard> {
    return this.getLocalBoard(difficulty);
  }

  // Get current settings
  getCurrentSettings(): any {
    return this.getSettings();
  }

  // Get current AI settings
  getCurrentAiSettings(): any {
    return this.getAiSettings();
  }

  // Helper method to check if a move is valid
  isValidMove(grid: number[][], row: number, col: number, num: number): boolean {
    // Check row
    for (let x = 0; x < 9; x++) {
      if (grid[row][x] === num) {
        return false;
      }
    }

    // Check column
    for (let y = 0; y < 9; y++) {
      if (grid[y][col] === num) {
        return false;
      }
    }

    // Check 3x3 box
    const startRow = Math.floor(row / 3) * 3;
    const startCol = Math.floor(col / 3) * 3;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (grid[startRow + i][startCol + j] === num) {
          return false;
        }
      }
    }

    return true;
  }

  // Method to check if the board is solved
  isSolved(grid: number[][]): boolean {
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        if (grid[i][j] === 0) {
          return false;
        }
      }
    }
    return true;
  }
}
