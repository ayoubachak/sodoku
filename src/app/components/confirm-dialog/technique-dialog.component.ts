import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MAT_DIALOG_DATA, MatDialogRef, MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { SudokuTechnique } from '../../services/technique-coach.service';

export interface TechniqueDialogData {
  title: string;
  message: string;
  technique: SudokuTechnique;
  value?: number;
  position?: { row: number, col: number };
  showRevealOption?: boolean;
}

@Component({
  selector: 'app-technique-dialog',
  standalone: true,
  imports: [CommonModule, MatDialogModule, MatButtonModule],
  template: `
    <h2 mat-dialog-title>{{ data.title }}</h2>
    <mat-dialog-content>
      <p [innerHTML]="data.message"></p>
      <div class="technique-details" *ngIf="data.position">
        <p class="position">
          Position: Row {{ data.position.row + 1 }}, Column {{ data.position.col + 1 }}
        </p>
        <p class="value" *ngIf="data.value">
          Value: {{ data.value }}
        </p>
      </div>
    </mat-dialog-content>
    <mat-dialog-actions>
      <button mat-button mat-dialog-close>Cancel</button>
      <button mat-button *ngIf="data.showRevealOption" [mat-dialog-close]="'reveal'">Reveal Value</button>
      <button mat-raised-button color="primary" [mat-dialog-close]="'apply'">Apply</button>
    </mat-dialog-actions>
  `,
  styles: [`
    .technique-details {
      margin: 16px 0;
      padding: 8px 16px;
      background-color: #f5f5f5;
      border-left: 4px solid #3f51b5;
      border-radius: 4px;
    }
    
    :host-context(.dark-theme) .technique-details {
      background-color: #333;
      border-left-color: #7986cb;
    }
    
    .position, .value {
      margin: 6px 0;
    }
    
    mat-dialog-actions {
      display: flex;
      justify-content: flex-end;
    }
  `]
})
export class TechniqueDialogComponent {
  constructor(
    public dialogRef: MatDialogRef<TechniqueDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: TechniqueDialogData
  ) {}
}