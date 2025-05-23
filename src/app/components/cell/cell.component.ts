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
  @Input() isOriginal: boolean = false;
  @Input() isSelected: boolean = false;
  @Input() isSameNumber: boolean = false;
  @Output() cellClick = new EventEmitter<void>();

  onClick(): void {
    if (!this.isOriginal) {
      this.cellClick.emit();
    }
  }
}
