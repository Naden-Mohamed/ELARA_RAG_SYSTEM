# تشغيل التطبيق الحقيقي

المسار الصحيح للواجهة هو `artifacts/maternal-ai-chat` وليس `mockup-sandbox`.

## Backend
```powershell
pnpm --filter @workspace/api-server dev
```

## Frontend (Terminal جديد)
```powershell
$env:PORT="5173"
$env:BASE_PATH="/"
pnpm --filter @workspace/maternal-ai-chat dev
```

افتحي `http://localhost:5173/`.

تمت إضافة Vite proxy بحيث `/api/*` من الواجهة يذهب إلى `http://localhost:3000/api/*`.

المصادر الموجودة داخل `attached_assets` هي المصادر المستخدمة في RAG، والأسئلة خارج نطاق المصادر يتم رفضها.
