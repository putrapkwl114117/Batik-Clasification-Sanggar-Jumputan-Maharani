from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
class ClassificationStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_classifications = db.Column(db.Integer, default=0)  
    
class ClassificationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Model untuk Produk Batik
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    images = db.relationship('ProductImage', backref='product', lazy=True)

class ProductImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
