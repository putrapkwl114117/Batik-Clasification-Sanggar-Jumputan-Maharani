from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from databases.mytables import db, Product,  ProductImage, ClassificationStats, ClassificationHistory
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import torch.nn as nn   
import io
import base64
from matplotlib.ticker import MaxNLocator
import numpy as np
import uuid
from datetime import datetime

app = Flask(__name__)

# app.secret_key = your scret key here
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://your databses here
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Folder untuk menyimpan gambar
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db.init_app(app)

with app.app_context():
    db.create_all()
    
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password123'


# Path file
model_save_path = "models/best_model_cnn.pth"
class_labels_path = "class_labels.json"

try:
    with open(class_labels_path, "r") as f:
        class_labels_list = json.load(f)  
        class_labels = {index: name for index, name in enumerate(class_labels_list)}
except FileNotFoundError:
    print(f"{class_labels_path} tidak ditemukan")
    class_labels = {}
except Exception as e:
    print(f"Error saat memuat class_labels: {str(e)}")
    class_labels = {}


# Inisialisasi model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='IMAGENET1K_V1')
num_classes = len(class_labels)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load model weights dengan weights_only=True
if os.path.exists(model_save_path):
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("Model berhasil dimuat")
    except Exception as e:
        print(f"Gagal memuat model: {str(e)}")
else:
    print(f"Model tidak ditemukan di {model_save_path}")

# Transformasi gambar
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

@app.route('/')
def index():
    mode = request.args.get('mode', 'single')
    products = Product.query.options(db.joinedload(Product.images)).all()

    categories = db.session.query(
        Product.category,
        db.func.min(ProductImage.image_path).label('first_image')
    ).join(ProductImage, Product.id == ProductImage.product_id)\
     .group_by(Product.category).all()

    return render_template('index.html', products=products, categories=categories, mode=mode)

@app.route('/clasifikasi')
def clasifikasi():
    stats = ClassificationStats.query.first()
    total_classifications = stats.total_classifications if stats else 0
    history = ClassificationHistory.query.order_by(ClassificationHistory.timestamp.desc()).all()
    history_data = [serialize_history(hist) for hist in history]
    class_count = {}
    for entry in history_data:
        class_name = entry['predicted_class']
        class_count[class_name] = class_count.get(class_name, 0) + 1
    products = Product.query.options(db.joinedload(Product.images)).all()

    # Query kategori unik beserta gambar pertama dari setiap kategori (sama seperti di index)
    categories = db.session.query(
        Product.category,
        db.func.min(ProductImage.image_path).label('first_image')
    ).join(ProductImage, Product.id == ProductImage.product_id)\
     .group_by(Product.category).all()
     
    return render_template(
        'clasifikasi.html',
        total_classifications=total_classifications,
        history=history_data,
        class_count=class_count,
        products=products,        
        categories=categories     
    )


def serialize_history(history):
    """Mengubah objek ClassificationHistory menjadi dictionary"""
    return {
        "timestamp": history.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": history.predicted_class,
    }


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "File tidak ditemukan!", 400

    file = request.files['file']
    if file.filename == '':
        return "Tidak ada file yang dipilih!", 400

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        image = Image.open(file_path).convert("RGB")
        image = transform_image(image)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        predicted_index = predicted.item()
        class_name = class_labels.get(predicted_index, "Label tidak ditemukan")
        hot_products = Product.query.filter_by(category=class_name).limit(4).all()
        history_entry = ClassificationHistory(image_path=file_path, predicted_class=class_name)
        db.session.add(history_entry)

        stats = ClassificationStats.query.first()
        if not stats:
            stats = ClassificationStats(total_classifications=1)
            db.session.add(stats)
        else:
            stats.total_classifications += 1
        db.session.commit()

        history_records = [
            {
                "timestamp": record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_class": record.predicted_class
            }
            for record in ClassificationHistory.query.order_by(ClassificationHistory.timestamp.desc()).all()
        ]

        return render_template(
        'clasifikasi.html',
        prediction=class_name,
        products=hot_products,
        total_classifications=stats.total_classifications,
        history=history_records,
        class_count={class_name: len(hot_products)}
    )



    except Exception as e:
        return f"Terjadi error: {str(e)}", 500

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['admin'] = True
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard_admin'))
    else:
        flash('Invalid username or password.', 'danger')
        return redirect(url_for('index'))


@app.route('/dashboard_admin')
def dashboard_admin():
    if not session.get('admin'):
        flash('You must be logged in to access the dashboard.', 'warning')
        return redirect(url_for('login'))

    try:
        products = Product.query.all()
        total_products = len(products)
        total_quantity = sum(product.quantity for product in products)  
        unique_categories = {product.category for product in products}
        total_categories = len(unique_categories)
        total_images = sum(len(product.images) for product in products)
        category_quantities = {}
        category_images = {}
        
        for product in products:
            if product.category not in category_quantities:
                category_quantities[product.category] = 0
                category_images[product.category] = 0
            
            category_quantities[product.category] += product.quantity  
            category_images[product.category] += len(product.images)

        fig1, ax1 = plt.subplots(figsize=(10, 6))

        categories = list(category_quantities.keys())
        quantities = list(category_quantities.values())
        images = list(category_images.values())

        bar_width = 0.2
        x = np.arange(len(categories))

        # Mengatur batang untuk setiap kategori
        bars1 = ax1.bar(x - bar_width, quantities, width=bar_width, label='Total Produk', color='b')
        bars2 = ax1.bar(x, quantities, width=bar_width, label='Total Stok', color='g')
        bars3 = ax1.bar(x + bar_width, [1]*len(categories), width=bar_width, label='Total Kategori', color='r') 
        bars4 = ax1.bar(x + 2*bar_width, images, width=bar_width, label='Total Gambar', color='y')

        ax1.set_title('Statistik Produk per Kategori')
        ax1.set_xlabel('Kategori')
        ax1.set_ylabel('Jumlah')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()

        ax1.yaxis.grid(True)

        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        chart_url1 = base64.b64encode(img1.getvalue()).decode('utf8')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        categories_list = list(category_images.keys())
        counts_list = list(category_images.values())
        pie_colors = plt.colormaps['Set2'](np.linspace(0, 1, len(categories_list)))

        wedges, texts, autotexts = ax2.pie(counts_list, labels=categories_list, autopct='%1.1f%%', startangle=140, colors=pie_colors)
        ax2.set_title('Distribusi Gambar per Kategori', fontsize=8)

        for autotext in autotexts:
            autotext.set_fontsize(6)
        ax2.legend(wedges, categories_list, title="Kategori", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        chart_url2 = base64.b64encode(img2.getvalue()).decode('utf8')
        plt.close(fig2)

        return render_template(
            'dashboard_admin.html',
            products=products,
            total_products=total_products,
            total_quantity=total_quantity,  
            total_categories=total_categories,
            total_images=total_images,
            chart_url1=chart_url1,
            chart_url2=chart_url2
        )

    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        app.logger.error(f"Error in dashboard_admin: {str(e)}")
        return redirect(url_for('login'))
    
    
@app.route('/get_products', methods=['GET'])
def get_products():
    try:
        products = Product.query.all()
        for product in products:
            product.image_path = None
            if product.images:  
                product.image_path = product.images[0].image_path

        return render_template('dashboard_admin.html', products=products)
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('dashboard_admin'))

# Edit Produk
@app.route('/edit_product/<int:product_id>', methods=['GET', 'POST'])
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)

    if request.method == 'POST':
        try:
            product.name = request.form.get('productName')
            product.type = request.form.get('productType')
            product.category = request.form.get('productCategory')
            product.price = float(request.form.get('productPrice'))
            product.quantity = int(request.form.get('productQuantity'))

            image_file = request.files.get('productImage')
            if image_file and image_file.filename != '':
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                product.image = filename

            db.session.commit()

            flash('Product updated successfully!', 'success')
            return redirect(url_for('dashboard_admin'))

        except Exception as e:
            flash(f"Error updating product: {str(e)}", 'danger')
            return redirect(url_for('dashboard_admin'))

    return render_template('edit_product.html', product=product)


@app.route('/add_product', methods=['POST'])
def add_product():
    try:
        # Ambil data dari form
        name = request.form.get('productName')
        type_ = request.form.get('productType')
        category = request.form.get('productCategory')
        price = float(request.form.get('productPrice'))
        quantity = int(request.form.get('productQuantity'))
        image_files = request.files.getlist('productImages')

        # Simpan produk baru
        new_product = Product(
            name=name,
            type=type_,
            category=category,
            price=price,
            quantity=quantity
        )
        db.session.add(new_product)
        db.session.commit()
        for image_file in image_files:
            if image_file and image_file.filename != '':
                unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
                image_file.save(image_path)
                image_path = image_path.replace('\\', '/')

                new_image = ProductImage(product_id=new_product.id, image_path=image_path)
                db.session.add(new_image)

        db.session.commit()

        flash("Product added successfully!", "success")
        return redirect(url_for('dashboard_admin'))
    except Exception as e:
        db.session.rollback()  
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('dashboard_admin'))


# Route untuk Menghapus Produk
@app.route('/delete_product/<int:product_id>', methods=['POST'])
def delete_product(product_id):
    try:
        # Query produk berdasarkan ID
        product = Product.query.get(product_id)
        if product:
            for image in product.images:
                image_path = image.image_path
                if os.path.exists(image_path):  
                    os.remove(image_path)  
                    print(f"Deleted image: {image_path}")  
                db.session.delete(image)
            db.session.delete(product)
            db.session.commit()

            flash('Product and associated images deleted successfully.', 'success')
        else:
            flash('Product not found.', 'danger')

        return redirect(url_for('dashboard_admin'))
    except Exception as e:
        db.session.rollback() 
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('dashboard_admin'))

@app.route('/logout')
def logout():
    session.pop('admin', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
