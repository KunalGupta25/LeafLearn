from app import app, db


from models import StudyPlan, StudyResource

def create_study_plans():
    with app.app_context():
        # Study Plan for Scores Below 50%
        low_score_plan = StudyPlan(
            user_id=1,  # Assuming user_id 1 for demonstration
            topic="Improving Programming Fundamentals",
            difficulty_level="Beginner"
        )
        db.session.add(low_score_plan)

        low_score_resources = [
            {
                "resource_type": "article",
                "title": "Understanding Object-Oriented Programming",
                "url": "https://example.com/oop-basics",
                "is_low_bandwidth": True
            },
            {
                "resource_type": "video",
                "title": "Python Lists vs Tuples",
                "url": "https://example.com/python-lists-vs-tuples",
                "is_low_bandwidth": True
            }
        ]

        for resource in low_score_resources:
            study_resource = StudyResource(
                study_plan=low_score_plan,
                resource_type=resource["resource_type"],
                title=resource["title"],
                url=resource["url"],
                is_low_bandwidth=resource["is_low_bandwidth"]
            )
            db.session.add(study_resource)

        # Study Plan for Scores Above 50%
        high_score_plan = StudyPlan(
            user_id=1,  # Assuming user_id 1 for demonstration
            topic="Advanced Programming Concepts",
            difficulty_level="Intermediate"
        )
        db.session.add(high_score_plan)

        high_score_resources = [
            {
                "resource_type": "article",
                "title": "Mastering REST APIs",
                "url": "https://example.com/rest-api-guide",
                "is_low_bandwidth": True
            },
            {
                "resource_type": "video",
                "title": "Understanding Git Branching",
                "url": "https://example.com/git-branching",
                "is_low_bandwidth": True
            }
        ]

        for resource in high_score_resources:
            study_resource = StudyResource(
                study_plan=high_score_plan,
                resource_type=resource["resource_type"],
                title=resource["title"],
                url=resource["url"],
                is_low_bandwidth=resource["is_low_bandwidth"]
            )
            db.session.add(study_resource)

        db.session.commit()
        print("Study plans created successfully!")

if __name__ == "__main__":
    create_study_plans()
