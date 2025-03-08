from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import login_user, login_required, logout_user, current_user
import torch
import re

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
    """
    Calculate similarity between user answer and correct answer based on keywords.
    text1: user's answer (long text paragraph)
    text2: correct answer containing keywords separated by |
    
    Returns a score between 0 and 1 based on:
    - Percentage of required keywords found in the answer
    - Bonuses for proper explanation and context
    - Penalties for very short or irrelevant answers
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        print(f"Invalid input types: text1 type = {type(text1)}, text2 type = {type(text2)}")
        return 0.0
    
    if not text1.strip() or not text2.strip():
        print("Empty text input")
        return 0.0
    
    # Preprocess the texts
    def preprocess_text(text):
        text = text.lower()
        text = ' '.join(text.split())
        import re
        text = re.sub(r'[^\w\s()\.+]', ' ', text)
        return text
    
    def find_keyword_matches(answer_text, keyword):
        """Find all variations of a keyword in the answer text with context awareness"""
        keyword = keyword.strip().lower()
        
        if keyword in ['oop', 'api', 'sql', 'xml', 'json', 'http']:
            return keyword in answer_text.lower().split()
        
        # Create variations of the keyword
        variations = [
            r'\b' + keyword + r'\b',  
            r'\b' + keyword + r's\b',  
            r'\b' + keyword + r'ing\b', 
            r'\b' + keyword + r'ed\b',  
            r'\b' + keyword + r'es\b',  
        ]
        
        
        if ' ' in keyword:
            parts = keyword.split()
            variations.extend([
                r'\b' + r'\s+'.join(parts) + r'\b',  
                r'\b' + r'\s*'.join(parts) + r'\b',  
                r'\b' + r'[\s-]*'.join(parts) + r'\b',  
            ])
            
            # Also check if all parts appear within a reasonable distance
            all_parts_present = all(part in answer_text for part in parts)
            if all_parts_present:
                # Check if parts appear within 10 words of each other
                words = answer_text.split()
                positions = []
                for part in parts:
                    for i, word in enumerate(words):
                        if part in word:
                            positions.append(i)
                            break
                if positions and max(positions) - min(positions) <= 10:
                    return True
        
        # Check if any variation matches
        import re
        for pattern in variations:
            if re.search(pattern, answer_text.lower()):
                return True
                
        return False
    
    def analyze_keyword_context(answer_text, keyword, window_size=10):
        """Analyze if the keyword is used in proper context"""
        words = answer_text.split()
        keyword_variations = [keyword] + [keyword + s for s in ['s', 'ing', 'ed', 'es']]
        
        for i, word in enumerate(words):
            if any(kw in word.lower() for kw in keyword_variations):
                # Get surrounding words
                start = max(0, i - window_size)
                end = min(len(words), i + window_size)
                context = ' '.join(words[start:end])
                
                # Check for explanation indicators
                explanation_indicators = ['is', 'means', 'refers to', 'involves', 'includes', 'represents', 
                                       'allows', 'enables', 'helps', 'used for', 'example']
                if any(indicator in context.lower() for indicator in explanation_indicators):
                    return True
        return False
    
    processed_answer = preprocess_text(text1)
    
    # Split the correct answer into keywords
    keywords = [kw.strip() for kw in text2.split('|')]
    total_keywords = len(keywords)
    
    print("\nAnalyzing answer:")
    print(f"Total keywords to match: {total_keywords}")
    print(f"Keywords: {keywords}")
    
    # Count matched keywords and analyze their context
    matched_keywords = []
    keywords_with_context = []
    
    for keyword in keywords:
        if find_keyword_matches(processed_answer, keyword):
            matched_keywords.append(keyword)
            if analyze_keyword_context(processed_answer, keyword):
                keywords_with_context.append(keyword)
    
    # Calculate base score from keyword matches
    matched_count = len(matched_keywords)
    context_count = len(keywords_with_context)
    
    # Base score is the percentage of keywords matched
    base_score = matched_count / total_keywords if total_keywords > 0 else 0.0
    
    print(f"\nMatched keywords ({matched_count}/{total_keywords}):")
    for kw in matched_keywords:
        print(f"- {kw}")
    
    print(f"\nKeywords with proper context ({context_count}/{matched_count}):")
    for kw in keywords_with_context:
        print(f"- {kw}")
    
    # Apply adjustments
    final_score = base_score
    
    # Bonus for keywords used in proper context
    context_bonus = (context_count / total_keywords) * 0.2 if total_keywords > 0 else 0.0  # Up to 20% bonus for good context
    final_score = min(1.0, final_score + context_bonus)
    
    # Bonus for comprehensive answers
    word_count = len(processed_answer.split())
    if word_count >= 100:  # Long, detailed answer
        final_score = min(1.0, final_score + 0.15)
        print("Applied length bonus: +0.15 for comprehensive answer")
    elif word_count >= 50:  # Medium length answer
        final_score = min(1.0, final_score + 0.1)
        print("Applied length bonus: +0.1 for good length answer")
    elif word_count < 20:  # Too short
        final_score = max(0.0, final_score - 0.1)
        print("Applied length penalty: -0.1 for short answer")
    
    # Bonus for well-structured answer
    structure_patterns = [
        r'\d\.',  
        r'[-•*]',  
        r'(?i)\b(first|second|third|finally|moreover|however)\b',  
        r'(?i)\b(for example|such as|specifically)\b',  
        r'(?i)\b(because|therefore|thus|hence)\b',  
    ]
    
    structure_matches = 0
    for pattern in structure_patterns:
        if re.search(pattern, text1):
            structure_matches += 1
    
    if structure_matches >= 2:
        structure_bonus = 0.1  # Up to 10% bonus for good structure
        final_score = min(1.0, final_score + structure_bonus)
        print(f"Applied structure bonus: +{structure_bonus} for well-structured answer")
    
    # Ensure final score is between 0 and 1
    final_score = max(0.0, min(1.0, final_score))
    
    print(f"\nScoring breakdown:")
    print(f"Base score (keyword matches): {base_score:.2f}")
    print(f"Context bonus: {context_bonus:.2f}")
    print(f"Final score: {final_score:.4f}")
    
    return final_score

def get_similarity_threshold(user_answer, correct_answer):
    """
    Determine the similarity threshold based on answer characteristics.
    Returns a lower threshold to allow for partial credit.
    """
    
    threshold = 0.3  # Lower base threshold for keyword-based matching
    
    
    keyword_count = len(correct_answer.split('|'))
    
    
    if keyword_count <= 3:
        threshold = 0.4  # Still strict but more lenient for few keywords
    elif keyword_count >= 8:
        threshold = 0.2  # More lenient for many keywords
 
    word_count = len(user_answer.split())
    if word_count < 10:
        threshold += 0.1  
    elif word_count > 50:
        threshold -= 0.1  
    
    return threshold

def generate_study_plan(user_id, quiz_results):
    # Analyze quiz results and create personalized study plan
    weak_areas = []
    for result in quiz_results:
        if result.score < 0.7:  
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
    try:
        data = request.json
        if not data:
            print("Error: No JSON data received")
            return jsonify({"error": "No data received"}), 400
            
        quiz_id = data.get('quiz_id')
        answers = data.get('answers')
        
        print(f"\nReceived quiz submission:")
        print(f"Quiz ID: {quiz_id}")
        print(f"Answers: {json.dumps(answers, indent=2)}")
        
        if not quiz_id or not answers:
            print("Error: Missing quiz_id or answers")
            return jsonify({"error": "Missing quiz_id or answers"}), 400
        
        # Validate quiz exists
        quiz = Quiz.query.get(quiz_id)
        if not quiz:
            print(f"Error: Quiz with id {quiz_id} not found")
            return jsonify({"error": f"Quiz with id {quiz_id} not found"}), 404
        
        print(f"\nProcessing quiz submission - quiz_id: {quiz_id}, answers count: {len(answers)}")
        
        quiz_result = QuizResult(
            user_id=current_user.id,
            quiz_id=quiz_id,
            score=0.0,
            completion_time=data.get('completion_time', 0)
        )
        db.session.add(quiz_result)
        
        total_questions = len(answers)
        total_score = 0.0
        processed_answers = []
        
        print("\nProcessing answers:")
        for answer in answers:
            try:
                question = Question.query.get(answer.get('question_id'))
                if not question:
                    print(f"Question not found for id: {answer.get('question_id')}")
                    continue
                
                user_answer = answer.get('answer', '').strip()
                correct_answer = question.correct_answer.strip()
                
                print(f"\nQuestion {question.id}:")
                print(f"Content: {question.content}")
                print(f"User answer: '{user_answer}'")
                print(f"Correct answer keywords: '{correct_answer}'")
                
                if not user_answer:
                    print("Empty user answer")
                    similarity_score = 0.0
                    is_correct = False
                else:
                    similarity_score = calculate_semantic_similarity(user_answer, correct_answer)
                    threshold = get_similarity_threshold(user_answer, correct_answer)
                    # Consider partially correct if score is above 30% of threshold
                    is_correct = similarity_score >= threshold
                    print(f"Similarity score: {similarity_score:.4f}")
                    print(f"Threshold: {threshold}")
                    print(f"Is correct: {is_correct}")
                
                # Add the similarity score to total (this gives partial credit)
                total_score += similarity_score
                
                quiz_answer = QuizAnswer(
                    quiz_result=quiz_result,
                    question_id=question.id,
                    user_answer=user_answer,
                    is_correct=is_correct,
                    similarity_score=similarity_score
                )
                db.session.add(quiz_answer)
                
                processed_answers.append({
                    'question_id': question.id,
                    'user_answer': user_answer,
                    'correct_answer': correct_answer,
                    'similarity_score': similarity_score,
                    'is_correct': is_correct,
                    'matched_keywords': [kw for kw in correct_answer.split('|') 
                                      if kw.strip().lower() in user_answer.lower()]
                })
                
            except Exception as e:
                print(f"Error processing answer: {str(e)}")
                import traceback
                print("Traceback:", traceback.format_exc())
                continue
        
        # Calculate percentage score based on average similarity scores
        if total_questions > 0:
            quiz_result.score = (total_score / total_questions) * 100 if total_questions > 0 else 0.0

            print(f"\nFinal score: {quiz_result.score:.1f}% (average similarity score: {total_score/total_questions:.2f})")
        
        try:
            db.session.commit()
            
            # Recommend study plan based on score
            if quiz_result.score < 50:
                study_plan = StudyPlan.query.filter_by(topic="Improving Programming Fundamentals", user_id=current_user.id).first()
            else:
                study_plan = StudyPlan.query.filter_by(topic="Advanced Programming Concepts", user_id=current_user.id).first()
            
            response_data = {
                'score': quiz_result.score,
                'correct_count': sum(1 for a in processed_answers if a['is_correct']),
                'total_questions': total_questions,
                'study_plan_id': study_plan.id,
                'detailed_results': processed_answers
            }
            
            print("\nSending response:", json.dumps(response_data, indent=2))
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error committing to database: {str(e)}")
            import traceback
            print("Traceback:", traceback.format_exc())
            db.session.rollback()
            return jsonify({"error": "Database error: " + str(e)}), 500
            
    except Exception as e:
        print(f"Unexpected error in submit_quiz: {str(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": "Server error: " + str(e)}), 500

@app.route('/toggle_dark_mode', methods=['POST'])
@login_required
def toggle_dark_mode():
    current_user.dark_mode = not current_user.dark_mode
    
    # Record sustainable action and update points
    if current_user.dark_mode:
        # Add points for enabling dark mode
        action = SustainableAction(
            user_id=current_user.id,
            action_type='dark_mode_enabled',
            points=10
        )
        db.session.add(action)
        current_user.sustainability_score += 10
    else:
        # Subtract points for disabling dark mode
        action = SustainableAction(
            user_id=current_user.id,
            action_type='dark_mode_disabled',
            points=-10
        )
        db.session.add(action)
        current_user.sustainability_score -= 10
    
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
@cache.cached(timeout=300, key_prefix=lambda: f'leaderboard_{request.args.get("time", "all")}_{current_user.dark_mode if current_user.is_authenticated else False}')  # Cache for 5 minutes with dark mode state
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
    if not data or 'response' not in data or 'correct_answer' not in data:
        return jsonify({"error": "Missing response or correct answer"}), 400
    
    result = calculate_semantic_similarity(data['response'], data['correct_answer'])
    return jsonify({"similarity_score": result})

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Load the AI model before starting the server
    if load_ai_model():
        print("AI model loaded successfully")
    else:
        print("Warning: AI model failed to load")
    
    # Run the app on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
