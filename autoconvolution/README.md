# Step Function Autoconvolution Visualizer

A React application that allows users to manipulate a step function on an interval [-1/4, 1/4] and visualize its autoconvolution.

## Features

- Adjust the number of pieces in the step function
- Select and modify individual pieces of the step function by clicking or dragging
- Real-time visualization of the autoconvolution function
- Comparison with Google's bound (1.5053)

## Getting Started

### Prerequisites

- Node.js (v14+ recommended)
- npm or yarn

### Installation

```bash
# Clone the repository (if applicable)
# git clone <repository-url>

# Navigate to the project directory
cd app

# Install dependencies
npm install
# or
yarn install
```

### Running the Application

```bash
# Start the development server
npm run dev
# or
yarn dev
```

The application will be available at [http://localhost:5173](http://localhost:5173).

### Building for Production

```bash
# Build the application
npm run build
# or
yarn build
```

The built files will be in the `dist` directory, ready to be deployed.

## Deploying to GitHub Pages (Jekyll site)

1. Build the application:
   ```bash
   npm run build
   ```

2. Copy the contents of the `dist` directory to your Jekyll site's appropriate location (usually within a subdirectory).

3. Update your Jekyll site's configuration if needed to properly serve the static files.

4. Commit and push the changes to your GitHub repository:
   ```bash
   git add .
   git commit -m "Add step function autoconvolution visualizer"
   git push
   ```

Your GitHub Pages site will automatically rebuild and deploy with the new content.

## Technologies Used

- React
- Recharts (for data visualization)
- TailwindCSS (for styling)
- Vite (for build tooling)