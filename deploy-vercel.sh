#!/bin/bash

# Vercel Deployment Script for Multimodal Design Analysis Suite
# This script helps automate the deployment process

set -e  # Exit on any error

echo "🚀 Starting Vercel Deployment Process..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    print_error "vercel.json not found. Please run this script from the project root."
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    print_warning "Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if user is logged in to Vercel
print_status "Checking Vercel authentication..."
if ! vercel whoami &> /dev/null; then
    print_warning "Not logged in to Vercel. Please log in..."
    vercel login
fi

# Navigate to frontend directory
print_status "Navigating to frontend directory..."
cd frontend

# Install dependencies
print_status "Installing frontend dependencies..."
npm ci

# Check if environment variables are set
print_status "Checking environment configuration..."
if [ ! -f ".env.local" ] && [ -z "$VITE_API_BASE_URL" ]; then
    print_warning "No environment variables found."
    echo "Please set VITE_API_BASE_URL either:"
    echo "1. Create a .env.local file with VITE_API_BASE_URL=your_backend_url"
    echo "2. Set it in Vercel dashboard after deployment"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the project locally to check for errors
print_status "Building project locally to check for errors..."
npm run build

if [ $? -eq 0 ]; then
    print_success "Local build successful!"
else
    print_error "Local build failed. Please fix errors before deploying."
    exit 1
fi

# Deploy to Vercel
print_status "Deploying to Vercel..."
vercel --prod

if [ $? -eq 0 ]; then
    print_success "Deployment successful! 🎉"
    echo ""
    echo "Next steps:"
    echo "1. Set environment variables in Vercel dashboard if not already done"
    echo "2. Test your deployed application"
    echo "3. Configure custom domain if needed"
    echo ""
    echo "Don't forget to:"
    echo "- Deploy your backend to Railway/Render/Heroku"
    echo "- Update CORS settings in your backend"
    echo "- Set up monitoring and analytics"
else
    print_error "Deployment failed. Check the error messages above."
    exit 1
fi

# Return to project root
cd ..

print_success "Deployment script completed!"
