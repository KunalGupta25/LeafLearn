from app import app, db
from models import Quiz, Question

def create_sample_quiz():
    with app.app_context():
        # Create programming fundamentals quiz
        quiz = Quiz(
            title="Programming Fundamentals",
            topic="Computer Science",
            difficulty="Intermediate"
        )
        db.session.add(quiz)
        
        # Create sample questions with keyword-based answers
        questions = [
            {
                "content": "Explain what Object-Oriented Programming (OOP) is and its main principles.",
                "correct_answer": "programming paradigm | objects | classes | inheritance | encapsulation | polymorphism | data | methods",
                "explanation": "The answer should mention that it's a programming paradigm using objects/classes and include key OOP principles.",
                "topic_area": "Programming Paradigms"
            },
            {
                "content": "What is the difference between a list and a tuple in Python?",
                "correct_answer": "mutable | immutable | square brackets | parentheses | ordered | modify | change | fixed",
                "explanation": "Key concepts include mutability vs immutability and their syntax differences.",
                "topic_area": "Data Structures"
            },
            {
                "content": "Explain what a REST API is and its key characteristics.",
                "correct_answer": "representational | state | transfer | http | stateless | client-server | uniform interface | resources | get | post | put | delete",
                "explanation": "The answer should cover the REST architecture style and HTTP methods.",
                "topic_area": "Web Development"
            },
            {
                "content": "What is the time complexity of binary search and how does it work?",
                "correct_answer": "O(log n) | divide | conquer | sorted | middle | half | comparison | efficient | binary",
                "explanation": "The answer should mention the time complexity and the divide-and-conquer approach.",
                "topic_area": "Algorithms"
            },
            {
                "content": "Explain what a closure is in programming.",
                "correct_answer": "function | inner | outer | scope | variables | access | remember | environment | nested",
                "explanation": "The answer should describe closures as functions that can access variables from their outer scope.",
                "topic_area": "Programming Concepts"
            },
            {
                "content": "What is the difference between synchronous and asynchronous programming?",
                "correct_answer": "blocking | non-blocking | sequential | parallel | wait | continue | concurrent | callbacks | promises",
                "explanation": "The answer should contrast blocking vs non-blocking execution and sequential vs parallel processing.",
                "topic_area": "Programming Concepts"
            },
            {
                "content": "Explain what Git branching is and its benefits.",
                "correct_answer": "version control | parallel | development | feature | main | merge | isolate | collaborate | workflow",
                "explanation": "The answer should cover branching as a feature of version control and its benefits for development.",
                "topic_area": "Version Control"
            },
            {
                "content": "What is dependency injection and why is it useful?",
                "correct_answer": "design pattern | coupling | dependencies | external | inject | modular | testing | flexibility | inversion",
                "explanation": "The answer should explain DI as a design pattern that reduces coupling and improves testing.",
                "topic_area": "Software Design"
            }
        ]
        
        for q_data in questions:
            question = Question(
                quiz=quiz,
                content=q_data["content"],
                correct_answer=q_data["correct_answer"],
                explanation=q_data["explanation"],
                topic_area=q_data["topic_area"]
            )
            db.session.add(question)
        
        db.session.commit()
        print("Programming quiz created successfully!")

if __name__ == "__main__":
    create_sample_quiz() 