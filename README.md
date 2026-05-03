# tugaspreprocessing-tugas3

Dapat disimpulkan bahwa dataset tersebut memiliki total 12 kolom, dengan jumlah maksimal baris untuk setiap kolom sebanyak 1000 baris. Akan tetapi, terdapat beberapa kolom yang memiliki jumlah baris kurang dari 1000, sehingga mengindikasikan adanya data yang hilang (missing values). Oleh karena itu, akan dilakukan proses identifikasi lebih lanjut terhadap kolom-kolom tersebut untuk mengetahui pola dan penanganan yang tepat.

Diperoleh bahwa kolom yang mengandung Missing Values (blanks/N/A dalam Python: NaN) adalah kolom StudentID, Name, Gender, AttendanceRate, StudyHoursPerWeek, PreviousGrade, ExtracurricularActivities, ParentalSupport, FinalGrade, Study Hours, Attendance (%), dan Online Classes Taken.

Jumlah data yang hilang pada masing-masing kolom bervariasi, di mana kolom StudyHoursPerWeek memiliki jumlah missing values terbanyak, sedangkan kolom ParentalSupport memiliki jumlah missing values paling sedikit. Hal ini menunjukkan bahwa hampir seluruh kolom dalam dataset masih memiliki data yang tidak lengkap, sehingga perlu dilakukan penanganan lebih lanjut sebelum proses analisis dilakukan.

HANDLING MISSING VALUE
Dalam Machine Learning, missing values merupakan masalah umum yang harus ditangani sebelum model dapat digunakan. Berdasarkan hasil eksplorasi data, diketahui bahwa hampir seluruh kolom dalam dataset memiliki nilai yang hilang (NaN), sehingga diperlukan penanganan yang tepat sesuai dengan jenis datanya.

Kolom StudentID merupakan identitas unik setiap siswa, sehingga tidak memiliki makna jika diisi dengan rata-rata atau median. Oleh karena itu, penanganan yang paling tepat adalah:

Menghapus baris yang memiliki missing values, atau Menghapus kolom jika tidak digunakan dalam analisis

