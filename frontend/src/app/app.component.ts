import { Component } from '@angular/core';
import {
  GeneratedPatient,
  VanillaGanService,
} from './services/vanilla-gan.service';

interface ArchLayer {
  name: string;
  detail: string;
}

interface MetricFigure {
  file: string;
  title: string;
  description: string;
}

type ViewPage =
  | 'introduction'
  | 'vanilla'
  | 'survival'
  | 'ctgan'
  | 'conclusion';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  selectedPage: ViewPage = 'introduction';

  readonly pages: { key: ViewPage; label: string }[] = [
    { key: 'introduction', label: 'Introduction' },
    { key: 'vanilla', label: 'Vanilla GAN Baseline' },
    { key: 'ctgan', label: 'CTGAN Baseline' },
    { key: 'survival', label: 'Survival GAN' },
    { key: 'conclusion', label: 'Conclusion' },
  ];

  readonly latentDim = 128;
  readonly hiddenWidth = 256;
  readonly epochs = 200;
  readonly batchSize = 500;
  readonly optimizer = 'Adam (lr=2e-4, betas=(0.5, 0.999))';
  readonly loss = 'Binary Cross Entropy (non saturating G loss)';
  readonly dataset = 'MSK-IMPACT 50k, 80/20 train/test';

  readonly generatorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'z ∈ ℝ^128 (Gaussian noise)' },
    { name: 'Linear', detail: '128 → 256' },
    { name: 'LeakyReLU', detail: 'negative slope = 0.2' },
    { name: 'Linear', detail: '256 → 256' },
    { name: 'LeakyReLU', detail: 'negative slope = 0.2' },
    { name: 'Linear', detail: '256 → 256' },
    { name: 'LeakyReLU', detail: 'negative slope = 0.2' },
    { name: 'Linear', detail: '256 → n_features' },
    { name: 'Output', detail: 'synthetic patient vector (standardised space)' },
  ];

  readonly discriminatorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'patient vector ∈ ℝ^n_features' },
    { name: 'Linear', detail: 'n_features → 256' },
    { name: 'LeakyReLU + Dropout', detail: 'slope = 0.2, p = 0.3' },
    { name: 'Linear', detail: '256 → 256' },
    { name: 'LeakyReLU + Dropout', detail: 'slope = 0.2, p = 0.3' },
    { name: 'Linear', detail: '256 → 1' },
    { name: 'Sigmoid', detail: 'P(real)' },
  ];

  readonly vanillaPerMethodFigures: MetricFigure[] = [
    {
      file: 'distributions.png',
      title: 'Feature Distributions',
      description:
        'Per feature histograms for real vs synthetic data. The bars should line up if the GAN learned that column.',
    },
    {
      file: 'km_curves.png',
      title: 'Kaplan–Meier Curves',
      description:
        'Kaplan–Meier curves for real and synthetic patients. Tells us if survival over time looks the same.',
    },
    {
      file: 'tsne_embedding.png',
      title: 't-SNE Embedding',
      description:
        '2D t-SNE projection of real and synthetic samples. Good overlap means they sit in roughly the same regions.',
    },
  ];

  readonly survivalGanDataset = 'MSK-IMPACT 50k survival';
  readonly survivalGanIter = 3000;
  readonly survivalGanBatchSize = 256;
  readonly survivalGanLatentDim = 128;
  readonly survivalGanGeneratorHiddenLayers = 3;
  readonly survivalGanGeneratorHiddenUnits = 250;
  readonly survivalGanGeneratorActivation = 'TANH';
  readonly survivalGanGeneratorResidual = 'Enabled';
  readonly survivalGanDiscriminatorHiddenLayers = 2;
  readonly survivalGanDiscriminatorHiddenUnits = 250;
  readonly survivalGanDiscriminatorActivation = 'LeakyReLU';
  readonly survivalGanDiscriminatorDropout = 0.1;
  readonly survivalGanOptimizer =
    'Adam (G: lr=1e-3, wd=1e-4, betas=(0.5,0.999); D: lr=1e-3, wd=1e-5, betas=(0.5,0.999))';
  readonly survivalGanLoss =
    'Wasserstein GAN with gradient penalty (lambda=10) + identifiability penalty (lambda=0.1)';

  readonly survivalGeneratorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'noise + conditional vector' },
    { name: 'Residual MLP Block 1', detail: 'Linear → TANH' },
    { name: 'Residual MLP Block 2', detail: 'Dropout(0.1) → Linear → TANH' },
    { name: 'Residual MLP Block 3', detail: 'Dropout(0.1) → Linear → TANH' },
    { name: 'Linear', detail: 'project to encoded tabular feature space' },
    {
      name: 'Mixed Activation Head',
      detail: 'softmax for discrete slices, identity for continuous slices',
    },
    {
      name: 'Output',
      detail: 'encoded synthetic survival row',
    },
  ];

  readonly survivalDiscriminatorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'encoded tabular row + conditional vector' },
    { name: 'MLP Block 1', detail: 'Dropout(0.1) → Linear → LeakyReLU' },
    { name: 'MLP Block 2', detail: 'Dropout(0.1) → Linear → LeakyReLU' },
    { name: 'MLP Block 3', detail: 'Dropout(0.1) → Linear → LeakyReLU' },
    { name: 'Linear', detail: 'project to critic score' },
    { name: 'Output', detail: 'Wasserstein critic value (no sigmoid)' },
  ];

  readonly survivalFigures: MetricFigure[] = [
    {
      file: 'distributions.png',
      title: 'Feature Distributions',
      description:
        'Per feature histograms for real vs synthetic data. The bars should line up if the GAN learned that column.',
    },
    {
      file: 'km_curves.png',
      title: 'Kaplan–Meier Curves',
      description:
        'Kaplan–Meier curves for real and synthetic patients. Tells us if survival over time looks the same.',
    },
    {
      file: 'tsne_embedding.png',
      title: 't-SNE Embedding',
      description:
        '2D t-SNE projection of real and synthetic samples. Good overlap means they sit in roughly the same regions.',
    },
  ];

  readonly ctganDataset = 'MSK-IMPACT 50k, 80/20 train/test';
  readonly ctganEpochs = 500;
  readonly ctganBatchSize = 500;
  readonly ctganLatentDim = 128;
  readonly ctganGeneratorDim = '(256, 256)';
  readonly ctganDiscriminatorDim = '(256, 256)';
  readonly ctganDiscriminatorSteps = 1;
  readonly ctganPac = 10;
  readonly ctganOptimizer = 'Adam (lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)';
  readonly ctganLoss = 'Wasserstein GAN with gradient penalty (lambda=10)';

  readonly ctganDiscreteColumns = [
    'Cancer Type',
    'Genetic Ancestry',
    'Disease Status',
    'FACETS QC',
    'MSI Type',
    'Sex',
    'Whole Genome Doubling Status (FACETS)',
    'Age at Diagnosis',
    'Mutation Count',
    'Sample coverage',
    'Number of Other Cancer Types',
    'status',
  ];

  readonly ctganGeneratorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'z ∈ ℝ^128 + conditional vector (mode specific)' },
    { name: 'Residual Block 1', detail: 'Linear → BatchNorm → ReLU (skip concat)' },
    { name: 'Residual Block 2', detail: 'Linear → BatchNorm → ReLU (skip concat)' },
    { name: 'Linear', detail: 'project to encoded tabular feature space' },
    {
      name: 'Mixed Activation Output',
      detail: 'tanh for continuous slices, Gumbel-softmax for discrete slices',
    },
  ];

  readonly ctganDiscriminatorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'pac=10 rows concatenated + conditional vector' },
    { name: 'Linear', detail: '(n_features × 10) → 256' },
    { name: 'LeakyReLU + Dropout', detail: 'slope = 0.2, p = 0.5' },
    { name: 'Linear', detail: '256 → 256' },
    { name: 'LeakyReLU + Dropout', detail: 'slope = 0.2, p = 0.5' },
    { name: 'Linear', detail: '256 → 1' },
    { name: 'Output', detail: 'Wasserstein critic value (no sigmoid)' },
  ];

  readonly ctganFigures: MetricFigure[] = [
    {
      file: 'distributions.png',
      title: 'Feature Distributions',
      description:
        'Per feature histograms for real vs synthetic data. The bars should line up if CTGAN learned that column.',
    },
    {
      file: 'km_curves.png',
      title: 'Kaplan–Meier Curves',
      description:
        'Kaplan–Meier curves for real and synthetic patients. Tells us if survival over time looks the same.',
    },
    {
      file: 'tsne_embedding.png',
      title: 't-SNE Embedding',
      description:
        '2D t-SNE projection of real and synthetic samples. Good overlap means they sit in roughly the same regions.',
    },
  ];

  generating = false;
  generated: GeneratedPatient | null = null;
  generateError: string | null = null;

  constructor(private vanillaGan: VanillaGanService) {}

  setPage(page: ViewPage): void {
    this.selectedPage = page;
  }

  generatePatient(): void {
    this.generating = true;
    this.generateError = null;
    this.vanillaGan.generate().subscribe({
      next: (patient) => {
        this.generated = patient;
        this.generating = false;
      },
      error: (err) => {
        this.generated = null;
        this.generateError =
          err?.message ??
          'Could not reach the GAN backend. Make sure backend/app.py is running on :5000.';
        this.generating = false;
      },
    });
  }

  formatValue(column: string, raw: number): string {
    if (Number.isInteger(raw)) {
      return raw.toString();
    }
    return raw.toFixed(3);
  }

  formatLatent(values: number[]): string {
    return values.map((v) => v.toFixed(3)).join(', ');
  }
}
