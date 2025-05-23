import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatSelectModule } from '@angular/material/select';

@Component({
  selector: 'app-menu',
  standalone: true,
  imports: [CommonModule, FormsModule, MatButtonModule, MatCardModule, MatSelectModule],
  templateUrl: './menu.component.html',
  styleUrl: './menu.component.css'
})
export class MenuComponent {
  selectedDifficulty: string = 'medium';
  difficulties: string[] = ['easy', 'medium', 'hard'];

  constructor(private router: Router) {}

  startGame(): void {
    // Navigate to game with the selected difficulty
    this.router.navigate(['/game'], { 
      queryParams: { difficulty: this.selectedDifficulty } 
    });
  }

  goToSettings(): void {
    this.router.navigate(['/settings']);
  }
}
