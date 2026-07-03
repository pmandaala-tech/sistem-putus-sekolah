from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import joblib
import os
import io
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'rahasia123'

# =============================================
# KONFIGURASI DATABASE MYSQL
# =============================================
import os

# Ganti bagian SQLALCHEMY_DATABASE_URI di app.py Anda dengan ini:
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://uxuu7ircb8ocviwa:7nF1bhRI13rhsCbwRsnZ@bqlvdhriqqy6xljt661s-mysql.services.clever-cloud.com:3306/bqlvdhriqqy6xljt661s'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =============================================
# LOAD MODEL MACHINE LEARNING
# =============================================
MODEL_PATH   = os.path.join('model', 'model_risiko.pkl')
ENCODER_PATH = os.path.join('model', 'label_encoder.pkl')

try:
    ml_model = joblib.load(MODEL_PATH)
    le       = joblib.load(ENCODER_PATH)
    print("✅ Model ML berhasil dimuat!")
except Exception as e:
    ml_model = None
    le       = None
    print(f"⚠️ Model ML gagal dimuat: {e}")

# =============================================
# SETUP FLASK-LOGIN
# =============================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Silakan login terlebih dahulu!'
login_manager.login_message_category = 'warning'

# =============================================
# MODEL DATABASE — Tabel Users
# =============================================
class UserModel(db.Model):
    __tablename__ = 'users'

    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    nama     = db.Column(db.String(100), nullable=False)
    role     = db.Column(db.String(20), nullable=False, default='guru_bk')
    password = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# =============================================
# MODEL DATABASE — Tabel Siswa
# =============================================
class Siswa(db.Model):
    __tablename__ = 'siswa'

    id            = db.Column(db.Integer, primary_key=True)
    nis           = db.Column(db.String(20), unique=True, nullable=False)
    nama          = db.Column(db.String(100), nullable=False)
    kelas         = db.Column(db.String(10), nullable=False)
    kehadiran     = db.Column(db.Float, nullable=False)
    nilai         = db.Column(db.Float, nullable=False)
    ekonomi       = db.Column(db.Integer, default=1)
    jarak_km      = db.Column(db.Float, default=0)
    ortu_terlibat = db.Column(db.Integer, default=1)
    mengulang     = db.Column(db.Integer, default=0)
    risiko        = db.Column(db.String(10), nullable=False, default='RENDAH')
    created_at    = db.Column(db.DateTime, default=datetime.now)
    updated_at    = db.Column(db.DateTime, default=datetime.now,
                              onupdate=datetime.now)

    def to_dict(self):
        return {
            'id'           : self.id,
            'nis'          : self.nis,
            'nama'         : self.nama,
            'kelas'        : self.kelas,
            'kehadiran'    : self.kehadiran,
            'nilai'        : self.nilai,
            'ekonomi'      : self.ekonomi,
            'jarak_km'     : self.jarak_km,
            'ortu_terlibat': self.ortu_terlibat,
            'mengulang'    : self.mengulang,
            'risiko'       : self.risiko,
        }

    def __repr__(self):
        return f'<Siswa {self.nama}>'

# =============================================
# MODEL USER UNTUK FLASK-LOGIN
# =============================================
class User(UserMixin):
    def __init__(self, id, username, nama, role):
        self.id       = id
        self.username = username
        self.nama     = nama
        self.role     = role

@login_manager.user_loader
def load_user(user_id):
    user_data = UserModel.query.get(int(user_id))
    if user_data:
        return User(user_data.id, user_data.username,
                    user_data.nama, user_data.role)
    return None

# =============================================
# DECORATOR CEK ROLE ADMIN
# =============================================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.role != 'admin':
            flash('Akses ditolak! Hanya Administrator yang bisa mengakses halaman ini.',
                  'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# =============================================
# FUNGSI PREDIKSI ML
# =============================================
def prediksi_risiko(kehadiran, nilai_rata, ekonomi,
                    jarak_km, ortu_terlibat, mengulang):
    if ml_model is not None and le is not None:
        input_data = np.array([[
            kehadiran, nilai_rata, ekonomi,
            jarak_km, ortu_terlibat, mengulang
        ]])

        prediksi_encoded = ml_model.predict(input_data)[0]
        risiko           = le.inverse_transform([prediksi_encoded])[0]

        proba      = ml_model.predict_proba(input_data)[0]
        proba_dict = {
            le.inverse_transform([i])[0]: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        }
        print(f"✅ Prediksi ML: {risiko} | Probabilitas: {proba_dict}")
        return risiko, proba_dict

    else:
        print("⚠️ Menggunakan logika sederhana (model ML tidak tersedia)")
        if kehadiran < 75 and nilai_rata < 65:
            risiko = 'TINGGI'
        elif kehadiran < 85 or nilai_rata < 70:
            risiko = 'SEDANG'
        else:
            risiko = 'RENDAH'
        return risiko, {}

# Jembatan fungsi agar route edit tidak menghasilkan error NameError
def prediksi_risiko_ml(kehadiran, nilai_rata, ekonomi, jarak_km, ortu_terlibat, mengulang):
    risiko, proba = prediksi_risiko(kehadiran, nilai_rata, ekonomi, jarak_km, ortu_terlibat, mengulang)
    return risiko, proba

# =============================================
# INISIALISASI DATABASE & DATA AWAL
# =============================================
def init_db():
    db.create_all()

    if not UserModel.query.filter_by(username='admin').first():
        users_default = [
            UserModel(username='admin', nama='Administrator',
                      role='admin',
                      password=generate_password_hash('admin123')),
            UserModel(username='bk', nama='Guru BK',
                      role='guru_bk',
                      password=generate_password_hash('bk123')),
            UserModel(username='kepsek', nama='Kepala Sekolah',
                      role='kepsek',
                      password=generate_password_hash('kepsek123')),
        ]
        db.session.add_all(users_default)
        db.session.commit()
        print('✅ Users default berhasil dibuat!')

    if not Siswa.query.first():
        siswa_default = [
            Siswa(nis='001', nama='Ahmad Fauzi',  kelas='X-A',
                  kehadiran=65, nilai=55, ekonomi=0,
                  jarak_km=12, ortu_terlibat=0, mengulang=1,
                  risiko='TINGGI'),
            Siswa(nis='002', nama='Budi Santoso', kelas='X-B',
                  kehadiran=90, nilai=80, ekonomi=2,
                  jarak_km=3,  ortu_terlibat=1, mengulang=0,
                  risiko='RENDAH'),
            Siswa(nis='003', nama='Citra Dewi',   kelas='XI-A',
                  kehadiran=75, nilai=65, ekonomi=1,
                  jarak_km=7,  ortu_terlibat=1, mengulang=0,
                  risiko='SEDANG'),
        ]
        db.session.add_all(siswa_default)
        db.session.commit()
        print('✅ Data siswa default berhasil dibuat!')

# =============================================
# ROUTE LOGIN
# =============================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username  = request.form['username']
        password  = request.form['password']
        user_data = UserModel.query.filter_by(username=username).first()

        if user_data and check_password_hash(user_data.password, password):
            user = User(user_data.id, user_data.username,
                        user_data.nama, user_data.role)
            login_user(user, remember=request.form.get('remember'))
            flash(f'Selamat datang, {user_data.nama}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))

        flash('Username atau password salah!', 'danger')

    return render_template('login.html')

# =============================================
# ROUTE LOGOUT
# =============================================
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Kamu berhasil logout.', 'info')
    return redirect(url_for('login'))

# =============================================
# ROUTE DASHBOARD
# =============================================
@app.route('/')
@login_required
def dashboard():
    total         = Siswa.query.count()
    risiko_tinggi = Siswa.query.filter_by(risiko='TINGGI').count()
    risiko_sedang = Siswa.query.filter_by(risiko='SEDANG').count()
    risiko_rendah = Siswa.query.filter_by(risiko='RENDAH').count()

    statistik = {
        'total_siswa'  : total,
        'risiko_tinggi': risiko_tinggi,
        'risiko_sedang': risiko_sedang,
        'risiko_rendah': risiko_rendah
    }

    prioritas = Siswa.query.filter_by(risiko='TINGGI').all()

    return render_template('index.html',
                           statistik=statistik,
                           prioritas=prioritas)

# =============================================
# ROUTE DAFTAR SISWA
# =============================================
@app.route('/siswa')
@login_required
def daftar_siswa():
    siswa = Siswa.query.order_by(Siswa.risiko.asc()).all()
    return render_template('siswa.html', siswa=siswa)

# =============================================
# ROUTE INPUT SISWA
# =============================================
@app.route('/input', methods=['GET', 'POST'])
@login_required
def input_siswa():
    if request.method == 'POST':
        nis = request.form['nis'].strip()

        if Siswa.query.filter_by(nis=nis).first():
            flash(f'NIS {nis} sudah terdaftar!', 'danger')
            return redirect(url_for('input_siswa'))

        kehadiran     = float(request.form['kehadiran'])
        nilai_rata    = float(request.form['nilai_rata'])
        ekonomi       = int(request.form['ekonomi'])
        jarak_km      = float(request.form['jarak_km'])
        ortu_terlibat = int(request.form['ortu_terlibat'])
        mengulang     = int(request.form['mengulang'])

        # Prediksi menggunakan Model ML
        risiko, proba_dict = prediksi_risiko(
            kehadiran, nilai_rata, ekonomi,
            jarak_km, ortu_terlibat, mengulang
        )

        # Simpan ke database MySQL
        siswa_baru = Siswa(
            nis           = nis,
            nama          = request.form['nama'].strip(),
            kelas         = request.form['kelas'],
            kehadiran     = kehadiran,
            nilai         = nilai_rata,
            ekonomi       = ekonomi,
            jarak_km      = jarak_km,
            ortu_terlibat = ortu_terlibat,
            mengulang     = mengulang,
            risiko        = risiko
        )
        db.session.add(siswa_baru)
        db.session.commit()

        flash(f"Data siswa {siswa_baru.nama} berhasil disimpan!", 'success')

        detail = siswa_baru.to_dict()
        detail['nilai_rata']   = nilai_rata
        detail['rekomendasi']  = get_rekomendasi(risiko)
        detail['probabilitas'] = proba_dict
        return render_template('detail.html', siswa=detail)

    return render_template('input.html')

# =============================================
# ROUTE DETAIL SISWA
# =============================================
@app.route('/detail/<nis>')
@login_required
def detail_siswa(nis):
    siswa = Siswa.query.filter_by(nis=nis).first()

    if siswa is None:
        flash(f'Siswa dengan NIS {nis} tidak ditemukan!', 'danger')
        return redirect(url_for('daftar_siswa'))

    detail = siswa.to_dict()
    detail['nilai_rata']   = siswa.nilai
    detail['rekomendasi']  = get_rekomendasi(siswa.risiko)
    detail['probabilitas'] = {}
    return render_template('detail.html', siswa=detail)

# =============================================
# ROUTE EDIT SISWA
# =============================================
@app.route('/siswa/edit/<nis>', methods=['GET', 'POST'])
@login_required
def edit_siswa(nis):
    siswa = Siswa.query.filter_by(nis=nis).first()

    if siswa is None:
        flash(f'Siswa dengan NIS {nis} tidak ditemukan!', 'danger')
        return redirect(url_for('daftar_siswa'))

    if request.method == 'POST':
        kehadiran     = float(request.form['kehadiran'])
        nilai_rata    = float(request.form['nilai_rata'])
        ekonomi       = int(request.form['ekonomi'])
        jarak_km      = float(request.form['jarak_km'])
        ortu_terlibat = int(request.form['ortu_terlibat'])
        mengulang     = int(request.form['mengulang'])

        # Prediksi ulang risiko dengan ML
        risiko, probabilitas = prediksi_risiko_ml(
            kehadiran, nilai_rata, ekonomi,
            jarak_km, ortu_terlibat, mengulang
        )

        # Update data siswa
        siswa.nama          = request.form['nama'].strip()
        siswa.kelas         = request.form['kelas']
        siswa.kehadiran     = kehadiran
        siswa.nilai         = nilai_rata
        siswa.ekonomi       = ekonomi
        siswa.jarak_km      = jarak_km
        siswa.ortu_terlibat = ortu_terlibat
        siswa.mengulang     = mengulang
        siswa.risiko        = risiko
        siswa.updated_at    = datetime.now()

        db.session.commit()

        flash(f'Data siswa {siswa.nama} berhasil diupdate! '
              f'Risiko: {risiko}', 'success')
        return redirect(url_for('detail_siswa', nis=nis))

    return render_template('edit_siswa.html', siswa=siswa)

# =============================================
# ROUTE HAPUS SISWA
# =============================================
@app.route('/siswa/hapus/<nis>', methods=['POST'])
@login_required
def hapus_siswa(nis):
    if current_user.role not in ['admin', 'guru_bk']:
        flash('Akses ditolak! Hanya Admin dan Guru BK yang bisa menghapus data siswa.', 'danger')
        return redirect(url_for('daftar_siswa'))

    siswa = Siswa.query.filter_by(nis=nis).first()

    if siswa is None:
        flash(f'Siswa dengan NIS {nis} tidak ditemukan!', 'danger')
        return redirect(url_for('daftar_siswa'))

    nama = siswa.nama
    db.session.delete(siswa)
    db.session.commit()

    flash(f'Data siswa {nama} berhasil dihapus!', 'success')
    return redirect(url_for('daftar_siswa'))

# =============================================
# ROUTE LAPORAN
# =============================================
@app.route('/laporan')
@login_required
def laporan():
    total         = Siswa.query.count()
    risiko_tinggi = Siswa.query.filter_by(risiko='TINGGI').count()
    risiko_sedang = Siswa.query.filter_by(risiko='SEDANG').count()
    risiko_rendah = Siswa.query.filter_by(risiko='RENDAH').count()

    kelas_list     = sorted(set(
        s.kelas for s in Siswa.query.with_entities(Siswa.kelas).distinct()
    ))
    data_per_kelas = []

    for kelas in kelas_list:
        siswa_kelas = Siswa.query.filter_by(kelas=kelas).all()
        if siswa_kelas:
            data_per_kelas.append({
                'kelas'         : kelas,
                'total'         : len(siswa_kelas),
                'tinggi'        : sum(1 for s in siswa_kelas if s.risiko == 'TINGGI'),
                'sedang'        : sum(1 for s in siswa_kelas if s.risiko == 'SEDANG'),
                'rendah'        : sum(1 for s in siswa_kelas if s.risiko == 'RENDAH'),
                'rata_kehadiran': round(
                    sum(s.kehadiran for s in siswa_kelas) / len(siswa_kelas), 1
                ),
                'rata_nilai'    : round(
                    sum(s.nilai for s in siswa_kelas) / len(siswa_kelas), 1
                ),
            })

    siswa_tinggi = Siswa.query.filter_by(risiko='TINGGI').all()

    return render_template('laporan.html',
        total          = total,
        risiko_tinggi  = risiko_tinggi,
        risiko_sedang  = risiko_sedang,
        risiko_rendah  = risiko_rendah,
        data_per_kelas = data_per_kelas,
        siswa_tinggi   = siswa_tinggi,
        kelas_list     = kelas_list,
    )

# =============================================
# ROUTE PENGATURAN
# =============================================
@app.route('/pengaturan')
@admin_required
def pengaturan():
    daftar_user = UserModel.query.all()
    total_user  = UserModel.query.count()
    return render_template('pengaturan.html',
                           daftar_user=daftar_user,
                           total_user=total_user)

# =============================================
# ROUTE TAMBAH USER
# =============================================
@app.route('/pengaturan/tambah-user', methods=['POST'])
@admin_required
def tambah_user():
    username = request.form['username'].strip()
    nama     = request.form['nama'].strip()
    role     = request.form['role']
    password = request.form['password']

    if not username or not nama or not password:
        flash('Semua field wajib diisi!', 'danger')
        return redirect(url_for('pengaturan'))

    if UserModel.query.filter_by(username=username).first():
        flash(f'Username "{username}" sudah digunakan!', 'danger')
        return redirect(url_for('pengaturan'))

    if len(password) < 6:
        flash('Password minimal 6 karakter!', 'danger')
        return redirect(url_for('pengaturan'))

    user_baru = UserModel(
        username = username,
        nama     = nama,
        role     = role,
        password = generate_password_hash(password)
    )
    db.session.add(user_baru)
    db.session.commit()

    flash(f'User "{username}" berhasil ditambahkan!', 'success')
    return redirect(url_for('pengaturan'))

# =============================================
# ROUTE EDIT USER
# =============================================
@app.route('/pengaturan/edit-user/<username>', methods=['POST'])
@admin_required
def edit_user(username):
    user = UserModel.query.filter_by(username=username).first()

    if not user:
        flash(f'User "{username}" tidak ditemukan!', 'danger')
        return redirect(url_for('pengaturan'))

    nama          = request.form['nama'].strip()
    role          = request.form['role']
    password_baru = request.form['password_baru'].strip()

    if not nama:
        flash('Nama tidak boleh kosong!', 'danger')
        return redirect(url_for('pengaturan'))

    user.nama = nama
    user.role = role

    if password_baru:
        if len(password_baru) < 6:
            flash('Password baru minimal 6 karakter!', 'danger')
            return redirect(url_for('pengaturan'))
        user.password = generate_password_hash(password_baru)
        flash(f'User "{username}" berhasil diupdate (termasuk password)!', 'success')
    else:
        flash(f'User "{username}" berhasil diupdate!', 'success')

    db.session.commit()
    return redirect(url_for('pengaturan'))

# =============================================
# ROUTE HAPUS USER
# =============================================
@app.route('/pengaturan/hapus-user/<username>', methods=['POST'])
@admin_required
def hapus_user(username):
    if username == current_user.username:
        flash('Tidak bisa menghapus akun yang sedang digunakan!', 'danger')
        return redirect(url_for('pengaturan'))

    user = UserModel.query.filter_by(username=username).first()

    if not user:
        flash(f'User "{username}" tidak ditemukan!', 'danger')
        return redirect(url_for('pengaturan'))

    nama = user.nama
    db.session.delete(user)
    db.session.commit()

    flash(f'User "{nama}" berhasil dihapus!', 'success')
    return redirect(url_for('pengaturan'))

# =============================================
# ROUTE IMPORT SISWA DARI EXCEL/CSV
# =============================================
@app.route('/siswa/import', methods=['GET', 'POST'])
@login_required
def import_siswa():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih!', 'danger')
            return redirect(url_for('import_siswa'))

        file = request.files['file']

        if file.filename == '':
            flash('Tidak ada file yang dipilih!', 'danger')
            return redirect(url_for('import_siswa'))

        if '.' not in file.filename:
            flash('File tidak memiliki ekstensi yang valid!', 'danger')
            return redirect(url_for('import_siswa'))

        ext = file.filename.rsplit('.', 1)[1].lower()
        allowed = {'xlsx', 'xls', 'csv'}

        if ext not in allowed:
            flash('Format file tidak didukung! Gunakan .xlsx, .xls, atau .csv', 'danger')
            return redirect(url_for('import_siswa'))

        try:
            if ext == 'csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            kolom_wajib = ['nis', 'nama', 'kelas', 'kehadiran',
                           'nilai', 'ekonomi', 'jarak_km',
                           'ortu_terlibat', 'mengulang']

            df.columns = df.columns.str.lower().str.strip()
            kolom_kurang = [k for k in kolom_wajib if k not in df.columns]

            if kolom_kurang:
                flash(f'Kolom berikut tidak ditemukan: {", ".join(kolom_kurang)}', 'danger')
                return redirect(url_for('import_siswa'))

            berhasil = 0
            gagal    = 0
            duplikat = 0
            errors   = []

            for index, row in df.iterrows():
                try:
                    if pd.api.types.is_number(row['nis']):
                        nis = str(int(row['nis'])).strip()
                    else:
                        nis = str(row['nis']).strip()

                    if Siswa.query.filter_by(nis=nis).first():
                        duplikat += 1
                        errors.append(f'Baris {index+2}: NIS {nis} sudah ada')
                        continue

                    kehadiran     = float(row['kehadiran'])
                    nilai         = float(row['nilai'])
                    ekonomi       = int(row['ekonomi'])
                    jarak_km      = float(row['jarak_km'])
                    ortu_terlibat = int(row['ortu_terlibat'])
                    mengulang     = int(row['mengulang'])

                    risiko, _ = prediksi_risiko_ml(
                        kehadiran, nilai, ekonomi,
                        jarak_km, ortu_terlibat, mengulang
                    )

                    siswa_baru = Siswa(
                        nis           = nis,
                        nama          = str(row['nama']).strip(),
                        kelas         = str(row['kelas']).strip(),
                        kehadiran     = kehadiran,
                        nilai         = nilai,
                        ekonomi       = ekonomi,
                        jarak_km      = jarak_km,
                        ortu_terlibat = ortu_terlibat,
                        mengulang     = mengulang,
                        risiko        = risiko
                    )
                    db.session.add(siswa_baru)
                    berhasil += 1

                except Exception as e:
                    gagal += 1
                    errors.append(f'Baris {index+2}: {str(e)}')
                    continue

            db.session.commit()

            if berhasil > 0:
                flash(f'✅ {berhasil} siswa berhasil diimport!', 'success')
            if duplikat > 0:
                flash(f'⚠️ {duplikat} siswa dilewati karena NIS sudah ada.', 'warning')
            if gagal > 0:
                flash(f'❌ {gagal} baris gagal diimport.', 'danger')

            return render_template('import_siswa.html', 
                                   errors=errors,
                                   berhasil=berhasil,
                                   duplikat=duplikat,
                                   gagal=gagal,
                                   selesai=True)

        except Exception as e:
            flash(f'Gagal membaca file: {str(e)}', 'danger')
            return redirect(url_for('import_siswa'))

    return render_template('import_siswa.html', selesai=False, errors=[])

# =============================================
# ROUTE DOWNLOAD TEMPLATE EXCEL
# =============================================
@app.route('/siswa/download-template')
@login_required
def download_template():
    data_contoh = {
        'nis'          : ['2025001', '2025002', '2025003'],
        'nama'         : ['Contoh Siswa 1', 'Contoh Siswa 2', 'Contoh Siswa 3'],
        'kelas'        : ['X-A', 'X-B', 'XI-A'],
        'kehadiran'    : [75, 60, 90],
        'nilai'        : [70, 55, 85],
        'ekonomi'      : [1, 0, 2],
        'jarak_km'     : [5, 15, 3],
        'ortu_terlibat': [1, 0, 1],
        'mengulang'    : [0, 1, 0],
    }

    df_template = pd.DataFrame(data_contoh)
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Data Siswa')
        worksheet = writer.sheets['Data Siswa']
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            worksheet.column_dimensions[col[0].column_letter].width = max_len + 5

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name='template_import_siswa.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# =============================================
# FUNGSI REKOMENDASI INTERVENSI
# =============================================
def get_rekomendasi(risiko):
    if risiko == 'TINGGI':
        return [
            'Hubungi orang tua segera dalam 1x24 jam',
            'Lakukan konseling individual minggu ini',
            'Koordinasi dengan wali kelas untuk pemantauan harian',
            'Ajukan ke program bantuan beasiswa jika masalah ekonomi',
            'Buat rencana intervensi tertulis dan pantau mingguan'
        ]
    elif risiko == 'SEDANG':
        return [
            'Jadwalkan konseling dalam minggu ini',
            'Pantau kehadiran setiap minggu',
            'Komunikasikan perkembangan kepada orang tua',
            'Berikan motivasi dan dukungan akademik'
        ]
    else:
        return [
            'Pantau kehadiran secara berkala setiap bulan',
            'Pertahankan komunikasi positif dengan siswa',
            'Lakukan check-in rutin tiap semester'
        ]

# =============================================
# JALANKAN APLIKASI
# =============================================
if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True)