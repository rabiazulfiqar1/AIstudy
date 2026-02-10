# AIstudy

An AI-powered study platform that helps students learn more effectively through intelligent video processing, project recommendations, study planning, and progress tracking.

## 🌟 Features

### 1. **Video Processing & Note Generation**
- **Video/Audio Upload**: Upload videos or audio files for AI-powered analysis
- **Automatic Summarization**: Get concise summaries of educational content
- **Smart Note Generation**: AI-generated notes from video/audio lectures
- **Translation**: Translate content to different languages
- **Transcript Export**: Extract and export video transcripts

### 2. **Intelligent Project Recommendations**
- **Topic-Based Recommendations**: Get project ideas tailored to your learning topic
- **Difficulty Tiers**: Projects matched to your skill level (beginner, intermediate, advanced)
- **Tech Stack Suggestions**: Recommended technologies for each project
- **Vector Similarity Search**: Semantic embeddings using pgvector + HuggingFace for smart matching
- **Interactive Project Cards**: Browse and interact with recommended projects

### 3. **Planner & Tracker**
- **Learning Roadmaps**: Create personalized study plans
- **Milestones & Goals**: Set and track learning objectives
- **Progress Charts**: Visualize your learning journey
- **Reminders**: Stay on track with study reminders

### 4. **Profile & Skills Management**
- **User Profiles**: Manage your learning profile
- **Skills Tracking**: Track skills you're learning and mastering
- **Project Interactions**: Save and interact with projects
- **Study Analytics**: View your learning statistics

## 🏗️ Architecture

This is a full-stack application with:
- **Frontend**: Next.js 15 with React 19, TypeScript, and Tailwind CSS
- **Backend**: FastAPI (Python) with async/await support
- **Database**: PostgreSQL with pgvector extension for semantic search
- **Authentication**: Supabase Auth
- **AI/ML**: HuggingFace models, Sentence Transformers, GROQ API

### Project Structure

```
AIstudy/
├── frontend/          # Next.js application
│   ├── src/
│   │   ├── app/      # Next.js pages and routes
│   │   ├── components/ # React components
│   │   └── lib/      # Utilities and configurations
│   └── package.json
├── backend/           # FastAPI application
│   ├── app/
│   │   ├── api/      # API routes
│   │   ├── core/     # Core configurations
│   │   ├── database/ # Database models and setup
│   │   ├── services/ # Business logic services
│   │   └── main.py   # FastAPI application entry
│   └── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Node.js 20+ and npm/yarn/pnpm
- Python 3.11+
- PostgreSQL with pgvector extension
- Supabase account (for authentication and storage)
- HuggingFace API token
- GROQ API key

### Environment Setup

1. **Backend Environment Variables**

Create a `.env` file in the `backend/` directory:

```bash
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
GITHUB_TOKEN=your_github_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

2. **Frontend Environment Variables**

Create a `.env.local` file in the `frontend/` directory:

```bash
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python -m app.database.init_db
```

5. Run the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
# or
pnpm install
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## 🔧 Technology Stack

### Frontend
- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **UI Library**: React 19
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI, shadcn/ui
- **State Management**: React Hooks
- **Authentication**: Supabase Auth
- **API Client**: Supabase JS

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Database**: PostgreSQL with SQLAlchemy (async)
- **Vector Search**: pgvector extension
- **AI/ML Libraries**: 
  - sentence-transformers
  - librosa (audio processing)
  - torch/torchvision
  - transformers (HuggingFace)
- **External APIs**: HuggingFace, GROQ
- **Authentication**: Supabase Auth
- **Video Processing**: yt-dlp, opencv-python

## 📝 API Documentation

Once the backend is running, you can access:
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health

### Main API Endpoints

- `GET /` - Root endpoint with welcome message
- `GET /health` - Health check with system status
- `POST /api/notes/upload` - Upload and process video/audio files
- `GET /api/recommendations/projects` - Get project recommendations
- `GET /api/profile` - User profile management
- Admin endpoints for data management

## 🎯 Usage

1. **Sign Up/Login**: Create an account or log in using Supabase authentication
2. **Upload Videos**: Go to the "Upload" page to process educational videos
3. **Get Notes**: Receive AI-generated summaries and notes
4. **Explore Projects**: Browse recommended projects based on your interests
5. **Plan Your Learning**: Use the planner to create study roadmaps
6. **Track Progress**: Monitor your learning progress and achievements

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is available under the MIT License.

## 🙏 Acknowledgments

- [Next.js](https://nextjs.org/) - The React Framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Supabase](https://supabase.com/) - Backend as a Service
- [HuggingFace](https://huggingface.co/) - AI models and transformers
- [Radix UI](https://www.radix-ui.com/) - UI component primitives
- [shadcn/ui](https://ui.shadcn.com/) - Beautifully designed components
