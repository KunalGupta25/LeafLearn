from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import login_user, login_required, logout_user, current_user
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datetime import datetime, timedelta
import json
from functools import wraps
from flask_caching import Cache

from extensions import db, login_manager
from models import User, Quiz, Question, QuizResult, QuizAnswer, SustainableAction, StudyPlan, StudyResource

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///learnleaf.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)

# Initialize cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize the AI model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = None

def load_ai_model():
    global model
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading model:", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully")
        # Test the model with a simple example
        test_text1 = "This is a test."
        test_text2 = "This is a test too."
        embeddings = model.encode([test_text1, test_text2])
        print("Model test successful - embeddings shape:", embeddings.shape)
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def calculate_semantic_similarity(text1, text2):
    global model
    if model is None:
        print("Error: Model not loaded")
        try:
            print("Attempting to reload model...")
            load_ai_model()
        except Exception as e:
            print(f"Failed to reload model: {str(e)}")
        return 0.0
    
    try:
        # Ensure texts are strings and not empty
        if not isinstance(text1, str) or not isinstance(text2, str):
            print(f"Invalid input types: text1 type = {type(text1)}, text2 type = {type(text2)}")
            return 0.0
        
        if not text1.strip() or not text2.strip():
            print("Empty text input")
            return 0.0
        
        print("\nCalculating similarity between:")
        print(f"Text 1: '{text1}'")
        print(f"Text 2: '{text2}'")
        
        # Get embeddings for both texts
        embeddings = model.encode([text1, text2])
        
        # Convert to PyTorch tensors
        import torch
        embedding1 = torch.FloatTensor(embeddings[0])
        embedding2 = torch.FloatTensor(embeddings[1])
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0), 
            embedding2.unsqueeze(0)
        )
        
        similarity_value = float(similarity[0])
        print(f"Calculated similarity: {similarity_value:.4f}")
        return similarity_value
        
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        return 0.0

def generate_study_plan(user_id, quiz_results):
    # Analyze quiz results and create personalized study plan
    weak_areas = []
    for result in quiz_results:
        if result.score < 0.7:  # Less than 70% correct
            weak_areas.append(result.quiz.topic)
    
    study_plan = StudyPlan(
        user_id=user_id,
        topic=", ".join(weak_areas),
        difficulty_level="intermediate"
    )
    db.session.add(study_plan)
    
    # Add recommended resources
    for topic in weak_areas:
        resource = StudyResource(
            study_plan=study_plan,
            resource_type="article",
            title=f"Understanding {topic}",
            url=f"/resources/{topic.lower().replace(' ', '-')}",
            is_low_bandwidth=True
        )
        db.session.add(resource)
    
    db.session.commit()
    return study_plan

@app.route('/')
def home():
    if current_user.is_authenticated:
        quiz_results = QuizResult.query.filter_by(user_id=current_user.id).all()
        study_plans = StudyPlan.query.filter_by(user_id=current_user.id).all()
        available_quizzes = Quiz.query.all()
        return render_template('dashboard.html',
                            quiz_results=quiz_results,
                            study_plans=study_plans,
                            available_quizzes=available_quizzes)
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        dark_mode = request.form.get('dark_mode') == 'on'
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email, dark_mode=dark_mode)
        user.set_password(password)
        db.session.add(user)
        
        if dark_mode:
            action = SustainableAction(
                user=user,
                action_type='dark_mode',
                points=10
            )
            db.session.add(action)
            user.sustainability_score += 10
        
        db.session.commit()
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    quiz_results = QuizResult.query.filter_by(user_id=current_user.id).all()
    study_plans = StudyPlan.query.filter_by(user_id=current_user.id).all()
    available_quizzes = Quiz.query.all()
    return render_template('dashboard.html', 
                         quiz_results=quiz_results, 
                         study_plans=study_plans,
                         available_quizzes=available_quizzes)

@app.route('/quizzes')
@login_required
def quizzes():
    quizzes = Quiz.query.all()
    return render_template('quizzes.html', quizzes=quizzes)

@app.route('/quiz/<int:quiz_id>')
@login_required
def take_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    return render_template('quiz.html', quiz=quiz)

@app.route('/submit_quiz', methods=['POST'])
@login_required
def submit_quiz():
    data = request.json
    quiz_id = data.get('quiz_id')
    answers = data.get('answers')
    
    if not quiz_id or not answers:
        return jsonify({"error": "Missing quiz_id or answers"}), 400
    
    print(f"\nReceived quiz submission - quiz_id: {quiz_id}, answers count: {len(answers)}")
    
    quiz_result = QuizResult(
        user_id=current_user.id,
        quiz_id=quiz_id,
        score=0.0,
        completion_time=data.get('completion_time', 0)
    )
    db.session.add(quiz_result)
    
    total_questions = len(answers)
    correct_count = 0
    
    # Similarity threshold for considering an answer correct
    SIMILARITY_THRESHOLD = 0.6  # Adjusted threshold to be more lenient
    
    print("\nProcessing answers:")
    for answer in answers:
        question = Question.query.get(answer['question_id'])
        if not question:
            print(f"Question not found for id: {answer['question_id']}")
            continue
            
        user_answer = answer['answer'].strip()
        correct_answer = question.correct_answer.strip()
        
        print(f"\nQuestion {question.id}:")
        print(f"Content: {question.content}")
        print(f"User answer: '{user_answer}'")
        print(f"Correct answer: '{correct_answer}'")
        
        if not user_answer:
            print("Empty user answer")
            similarity_score = 0.0
            is_correct = False
        else:
            similarity_score = calculate_semantic_similarity(user_answer, correct_answer)
            is_correct = similarity_score >= SIMILARITY_THRESHOLD
            print(f"Similarity score: {similarity_score:.4f}")
            print(f"Threshold: {SIMILARITY_THRESHOLD}")
            print(f"Is correct: {is_correct}")
        
        if is_correct:
            correct_count += 1
        
        quiz_answer = QuizAnswer(
            quiz_result=quiz_result,
            question_id=question.id,
            user_answer=answer['answer'],
            is_correct=is_correct
        )
        db.session.add(quiz_answer)
    
    # Calculate percentage score
    if total_questions > 0:
        quiz_result.score = (correct_count / total_questions) * 100
        print(f"\nFinal score: {quiz_result.score:.1f}% ({correct_count}/{total_questions} correct)")
    
    # Add sustainability points for using low bandwidth mode
    if data.get('low_bandwidth'):
        action = SustainableAction(
            user_id=current_user.id,
            action_type='low_bandwidth',
            points=5
        )
        db.session.add(action)
        current_user.sustainability_score += 5
    
    try:
        db.session.commit()
        study_plan = generate_study_plan(current_user.id, [quiz_result])
        
        return jsonify({
            'score': quiz_result.score,
            'correct_count': correct_count,
            'total_questions': total_questions,
            'study_plan_id': study_plan.id,
            'detailed_results': [{
                'question_id': answer['question_id'],
                'user_answer': answer['answer'].strip(),
                'correct_answer': Question.query.get(answer['question_id']).correct_answer.strip() if Question.query.get(answer['question_id']) else '',
                'similarity_score': calculate_semantic_similarity(
                    answer['answer'].strip(),
                    Question.query.get(answer['question_id']).correct_answer.strip()
                ) if Question.query.get(answer['question_id']) and answer['answer'].strip() else 0.0
            } for answer in answers]
        })
    except Exception as e:
        print(f"Error committing quiz result: {str(e)}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/toggle_dark_mode', methods=['POST'])
@login_required
def toggle_dark_mode():
    current_user.dark_mode = not current_user.dark_mode
    
    # Record sustainable action
    if current_user.dark_mode:
        action = SustainableAction(
            user_id=current_user.id,
            action_type='dark_mode',
            points=10
        )
        db.session.add(action)
        current_user.sustainability_score += 10
    
    db.session.commit()
    return jsonify({'dark_mode': current_user.dark_mode})

def calculate_user_stats(user):
    """Calculate detailed statistics for a user."""
    total_quizzes = len(user.quizzes)
    avg_score = sum(result.score for result in user.quizzes) / total_quizzes if total_quizzes > 0 else 0
    total_actions = len(user.sustainable_actions)
    
    # Calculate achievements
    achievements = {
        'quiz_master': total_quizzes >= 10,
        'eco_warrior': total_actions >= 20,
        'perfect_score': any(result.score == 100 for result in user.quizzes),
        'consistent_learner': total_quizzes >= 5 and avg_score >= 80
    }
    
    return {
        'total_quizzes': total_quizzes,
        'avg_score': avg_score,
        'total_actions': total_actions,
        'achievements': achievements
    }

@app.route('/leaderboard')
@cache.cached(timeout=300)  # Cache for 5 minutes
def leaderboard():
    # Get filter parameters
    time_filter = request.args.get('time', 'all')  # all, week, month
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Base query
    query = User.query
    
    # Apply time filter
    if time_filter == 'week':
        week_ago = datetime.utcnow() - timedelta(days=7)
        query = query.join(SustainableAction).filter(SustainableAction.created_at >= week_ago)
    elif time_filter == 'month':
        month_ago = datetime.utcnow() - timedelta(days=30)
        query = query.join(SustainableAction).filter(SustainableAction.created_at >= month_ago)
    
    # Get users ordered by sustainability score with pagination
    paginated_users = query.order_by(User.sustainability_score.desc())\
                          .paginate(page=page, per_page=per_page, error_out=False)
    
    # Calculate rank and stats for each user
    users_data = []
    start_rank = (page - 1) * per_page + 1
    
    for idx, user in enumerate(paginated_users.items):
        rank = start_rank + idx
        stats = calculate_user_stats(user)
        
        users_data.append({
            'user': user,
            'rank': rank,
            'stats': stats
        })
    
    # Get top 3 users of all time (for hall of fame)
    top_users = User.query.order_by(User.sustainability_score.desc()).limit(3).all()
    
    return render_template('leaderboard.html',
                         users_data=users_data,
                         top_users=top_users,
                         pagination=paginated_users,
                         time_filter=time_filter,
                         total_users=User.query.count())

@app.route('/api/leaderboard/stats')
@cache.cached(timeout=300)  # Cache for 5 minutes
def leaderboard_stats():
    """API endpoint for leaderboard statistics."""
    total_users = User.query.count()
    total_actions = SustainableAction.query.count()
    avg_score = db.session.query(db.func.avg(User.sustainability_score)).scalar() or 0
    
    return jsonify({
        'total_users': total_users,
        'total_actions': total_actions,
        'average_score': round(avg_score, 2),
        'last_updated': datetime.utcnow().isoformat()
    })

@app.route('/study_plan/<int:plan_id>')
@login_required
def view_study_plan(plan_id):
    plan = StudyPlan.query.get_or_404(plan_id)
    if plan.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    return render_template('study_plan.html', plan=plan)

@app.route('/update_resource_status', methods=['POST'])
@login_required
def update_resource_status():
    data = request.json
    resource = StudyResource.query.get_or_404(data['resource_id'])
    
    if resource.study_plan.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    
    resource.completion_status = data['completed']
    db.session.commit()
    return jsonify({"success": True})

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    data = request.json
    if not data or 'response' not in data:
        return jsonify({"error": "No quiz response provided"}), 400
    
    result = calculate_semantic_similarity(data['response'], data['response'])
    return jsonify(result)

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Load the AI model before starting the server
    if load_ai_model():
        print("AI model loaded successfully")
    else:
        print("Warning: AI model failed to load")
    
    app.run(debug=True) 