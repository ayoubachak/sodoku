import { TestBed } from '@angular/core/testing';

import { TechniqueCoachService } from './technique-coach.service';

describe('TechniqueCoachService', () => {
  let service: TechniqueCoachService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TechniqueCoachService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
