import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-cell',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './cell.component.html',
  styleUrl: './cell.component.css'
})
export class CellComponent {
  @Input() value: number = 0;
  @Input() notes: number[] = [];
  @Input() isOriginal: boolean = false;
  @Input() isSelected: boolean = false;
  @Input() isSameNumber: boolean = false;
  @Input() isConflict: boolean = false;
  @Input() isHighlighted: boolean = false;
  @Output() cellClick = new EventEmitter<void>();

  onClick(): void {
    this.cellClick.emit();
  }

  // Helper method to get notes for display in a 3x3 grid
  getNotesGrid(): (number | null)[][] {
    const grid: (number | null)[][] = [
      [null, null, null],
      [null, null, null], 
      [null, null, null]
    ];
    
    this.notes.forEach(note => {
      const row = Math.floor((note - 1) / 3);
      const col = (note - 1) % 3;
      grid[row][col] = note;
    });
    
    return grid;
  }
}
