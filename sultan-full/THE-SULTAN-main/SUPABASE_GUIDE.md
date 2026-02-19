# 🗄️ THE SULTAN - دليل Supabase Integration

## 📋 نظرة عامة

تم إضافة Supabase لتوفير:
- ✅ حفظ تاريخ المحادثات
- ✅ استرجاع المحادثات السابقة
- ✅ مشاركة المحادثات (اختياري)
- ✅ مصادقة المستخدمين (اختياري)

---

## 🚀 خطوات الإعداد

### 1. إنشاء مشروع Supabase

```bash
# زر https://supabase.com
# 1. سجل دخول أو أنشئ حساب
# 2. اضغط "New Project"
# 3. اختر اسم المشروع والباسورد
# 4. اختر المنطقة (الأقرب لك)
```

### 2. تطبيق Database Schema

```sql
# في Supabase Dashboard:
# 1. اذهب إلى SQL Editor
# 2. انسخ محتوى supabase/schema.sql
# 3. نفذ SQL
```

أو استخدم الملف:
```bash
supabase/schema.sql
```

### 3. الحصول على API Keys

```bash
# في Supabase Dashboard > Settings > API:
# 1. Project URL:       https://xxxxx.supabase.co
# 2. anon/public key:   eyJhbGc...
# 3. service_role key:  eyJhbGc... (للعمليات الخادمة فقط)
```

### 4. إضافة Environment Variables

أضف في Railway Dashboard أو ملف `.env.local`:

```env
SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
```

---

## 📁 الملفات المضافة

### 1. `lib/supabase.ts`
```typescript
// Supabase client configuration
// Database types للـ TypeScript
```

### 2. `app/api/chat/route.ts`
```typescript
// المسار الرئيسي مدمج مع Supabase
// GET: استرجاع المحادثات أو الرسائل
// POST: إرسال رسالة + حفظ (مع header x-conversation-id)
```

### 3. `components/ConversationsSidebar.tsx`
```typescript
// Sidebar لعرض المحادثات السابقة
// إمكانية الحذف
// اختيار محادثة سابقة
```

### 4. `supabase/schema.sql`
```sql
// Database schema
// Tables: conversations, messages
// RLS policies
// Indexes للأداء
```

---

## 🔄 كيفية الاستخدام

### كيف يعمل الآن؟
- المسار الرئيسي `app/api/chat/route.ts` مدمج مع Supabase تلقائياً.
- إذا كانت مفاتيح Supabase غير مضبوطة، سيعمل الرد بدون تخزين المحادثات.
- عند تفعيل المفاتيح، يتم حفظ المحادثات وإرجاع `x-conversation-id` لكل إنشاء جديد.

### تفعيل الحفظ عبر Supabase
1. أضف متغيرات البيئة في Railway أو `.env.local` (انظر أعلاه).
2. طبّق `supabase/schema.sql` في مشروع Supabase.
3. افتح الواجهة؛ سترى الشريط الجانبي للمحادثات يعمل فوراً بالحفظ/التحميل.

---

## 🗃️ Database Schema

### جدول `conversations`
```sql
- id:          UUID (primary key)
- title:       TEXT (عنوان المحادثة)
- user_id:     UUID (اختياري - للمصادقة)
- created_at:  TIMESTAMP
- updated_at:  TIMESTAMP
```

### جدول `messages`
```sql
- id:              UUID (primary key)
- conversation_id: UUID (foreign key)
- role:            TEXT (user/assistant/system)
- content:         TEXT (محتوى الرسالة)
- user_id:         UUID (اختياري)
- created_at:      TIMESTAMP
```

---

## 🔒 الأمان (RLS Policies)

```sql
# السياسات الحالية تسمح:
- ✅ قراءة المحادثات للجميع (anonymous)
- ✅ إضافة محادثات للجميع
- ✅ المستخدمين المصادقين يرون محادثاتهم فقط

# لتفعيل Auth فقط:
# احذف policies الـ anonymous
# أبق policies المستخدمين فقط
```

---

## 🎯 المميزات الإضافية

### 1. حفظ المحادثات تلقائياً
```typescript
// في route.ts:
saveConversation(messages, conversationId);
```

### 2. استرجاع المحادثات السابقة
```typescript
// GET /api/chat?conversationId=xxx
const { data } = await supabase
  .from('messages')
  .select('*')
  .eq('conversation_id', id);
```

### 3. عرض قائمة المحادثات
```typescript
// ConversationsSidebar component
<ConversationsSidebar 
  onSelectConversation={setConversationId}
  currentConversationId={conversationId}
/>
```

### 4. حذف المحادثات
```typescript
await supabase
  .from('conversations')
  .delete()
  .eq('id', conversationId);
```

---

## 🔧 التخصيص

### تفعيل المصادقة (Authentication)

```typescript
// 1. في Supabase Dashboard > Authentication
// فعّل Email/Password أو OAuth providers

// 2. أضف Auth UI:
import { Auth } from '@supabase/auth-ui-react';

<Auth
  supabaseClient={supabase}
  appearance={{ theme: ThemeSupa }}
  providers={['google', 'github']}
/>

// 3. احفظ user_id مع المحادثات:
const { data: { user } } = await supabase.auth.getUser();
await supabase.from('conversations').insert({
  title,
  user_id: user?.id
});
```

### تفعيل المشاركة (Sharing)

```sql
-- أضف عمود للمحادثات:
ALTER TABLE conversations ADD COLUMN is_public BOOLEAN DEFAULT false;

-- Policy جديد:
CREATE POLICY "Public conversations readable" ON conversations
  FOR SELECT USING (is_public = true);
```

---

## 📊 المراقبة والتحليلات

```typescript
// يمكنك إضافة:
// 1. عدد المحادثات لكل مستخدم
// 2. متوسط طول المحادثات
// 3. أكثر الأوقات استخداماً

// مثال:
SELECT 
  COUNT(*) as total_conversations,
  AVG(message_count) as avg_messages
FROM conversations;
```

---

## ⚡ الأداء

### Indexes المضافة
```sql
- idx_conversations_user_id
- idx_conversations_updated_at
- idx_messages_conversation_id
- idx_messages_created_at
```

### التحسينات
- ✅ استعلامات محسنة بـ indexes
- ✅ Row Level Security للأمان
- ✅ Cascade delete لحذف الرسائل تلقائياً

---

## 🐛 Troubleshooting

### مشكلة: لا تظهر المحادثات
```bash
# تحقق من:
1. Environment variables صحيحة
2. RLS policies مفعلة
3. Schema مطبق بشكل صحيح
```

### مشكلة: خطأ في الحفظ
```bash
# تحقق من:
1. anon key له صلاحيات الكتابة
2. RLS policies تسمح بـ INSERT
3. Foreign keys صحيحة
```

---

## 📝 الملاحظات

1. **الإصدار الحالي بدون Supabase لا يزال يعمل بشكل كامل**
2. Supabase اختياري تماماً - يمكن تفعيله لاحقاً
3. البيانات محفوظة بشكل آمن مع RLS
4. يمكن إضافة المصادقة لاحقاً بسهولة

---

## 🚀 الخطوة التالية

اختر إحدى الطرق:

### A. استخدام Supabase الآن:
```bash
1. أنشئ مشروع Supabase
2. طبق schema.sql
3. أضف environment variables
4. فعّل الملفات الجديدة
5. استمتع بتاريخ المحادثات!
```

### B. البقاء بدون Supabase:
```bash
# لا تفعل شيء - التطبيق يعمل كما كان
# الملفات الجديدة موجودة للمستقبل فقط
```

---

**الحالة:** ✅ جاهز للاستخدام في كلا الحالتين!
