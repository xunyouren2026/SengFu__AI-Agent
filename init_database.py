#!/usr/bin/env python3
"""
AGI Unified Framework - 数据库初始化脚本（修复版）

使用方法:
    python init_database.py

这个脚本会:
1. 检查并创建所有必要的数据库表
2. 插入初始模拟数据（如果不存在）
3. 确保后端API可以正常工作
"""

import os
import sys
import random
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def init_database():
    """初始化数据库"""
    print("=" * 60)
    print("AGI Unified Framework - 数据库初始化")
    print("=" * 60)
    
    try:
        # 导入数据库模块
        from database.models import Base, Model, User, Conversation, Message, Workflow
        from sqlalchemy import create_engine, text, inspect
        from sqlalchemy.orm import sessionmaker
        
        # 数据库路径
        db_path = os.path.join(project_root, 'data', 'app.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        print(f"\n数据库路径: {db_path}")
        
        # 创建引擎
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # 使用检查器来安全地创建表
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        print(f"\n已存在的表: {existing_tables}")
        
        # 创建所有表（使用 checkfirst=True 会自动跳过已存在的表）
        print("\n检查并创建数据库表...")
        
        # 逐个创建表，忽略已存在的
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                try:
                    table.create(engine, checkfirst=True)
                    print(f"  ✓ 创建表: {table.name}")
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"  ℹ️  表已存在: {table.name}")
                    else:
                        print(f"  ⚠️  创建表 {table.name} 时警告: {e}")
            else:
                print(f"  ℹ️  表已存在: {table.name}")
        
        # 创建索引（忽略已存在的）
        for table in Base.metadata.sorted_tables:
            for index in table.indexes:
                try:
                    index.create(engine, checkfirst=True)
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"  ⚠️  索引创建警告: {e}")
        
        print("\n✓ 数据库表检查完成")
        
        # 创建会话
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # 插入初始数据
        print("\n插入初始模拟数据...")
        
        # 1. 创建用户
        try:
            if session.query(User).count() == 0:
                admin = User(
                    username='admin',
                    email='admin@agi.local',
                    is_active=True,
                    is_admin=True,
                    created_at=datetime.now()
                )
                session.add(admin)
                session.commit()
                print("✓ 创建默认用户")
            else:
                print("ℹ️  用户已存在")
        except Exception as e:
            session.rollback()
            print(f"⚠️  用户创建跳过: {e}")
        
        # 2. 创建模型
        try:
            if session.query(Model).count() == 0:
                models = [
                    Model(
                        name='gpt-4',
                        display_name='AGI-Ultra',
                        provider='openai',
                        model_type='llm',
                        status='active',
                        max_tokens=8192,
                        cost_per_1k_input_tokens=0.03,
                        cost_per_1k_output_tokens=0.06,
                        avg_latency_ms=180,
                        success_rate=0.999,
                        total_requests=random.randint(10000, 50000),
                        capabilities=['text', 'code', 'reasoning'],
                        created_at=datetime.now()
                    ),
                    Model(
                        name='claude-3-opus',
                        display_name='AGI-Sage',
                        provider='anthropic',
                        model_type='llm',
                        status='active',
                        max_tokens=200000,
                        cost_per_1k_input_tokens=0.015,
                        cost_per_1k_output_tokens=0.075,
                        avg_latency_ms=220,
                        success_rate=0.998,
                        total_requests=random.randint(8000, 40000),
                        capabilities=['text', 'code', 'long-context'],
                        created_at=datetime.now()
                    ),
                    Model(
                        name='gemini-pro',
                        display_name='AGI-Flash',
                        provider='google',
                        model_type='llm',
                        status='active',
                        max_tokens=32768,
                        cost_per_1k_input_tokens=0.0005,
                        cost_per_1k_output_tokens=0.0015,
                        avg_latency_ms=150,
                        success_rate=0.995,
                        total_requests=random.randint(5000, 30000),
                        capabilities=['text', 'multimodal'],
                        created_at=datetime.now()
                    ),
                    Model(
                        name='qwen-72b',
                        display_name='Qwen-72B',
                        provider='alibaba',
                        model_type='llm',
                        status='active',
                        max_tokens=32768,
                        cost_per_1k_input_tokens=0.001,
                        cost_per_1k_output_tokens=0.002,
                        avg_latency_ms=200,
                        success_rate=0.997,
                        total_requests=random.randint(3000, 20000),
                        capabilities=['text', 'chinese', 'code'],
                        created_at=datetime.now()
                    ),
                ]
                session.add_all(models)
                session.commit()
                print(f"✓ 创建 {len(models)} 个模型")
            else:
                print("ℹ️  模型已存在")
        except Exception as e:
            session.rollback()
            print(f"⚠️  模型创建跳过: {e}")
        
        # 3. 创建工作流
        try:
            if session.query(Workflow).count() == 0:
                workflows = [
                    Workflow(
                        name='文本生成工作流',
                        description='自动生成文本内容的工作流',
                        status='active',
                        total_executions=random.randint(100, 1000),
                        successful_executions=random.randint(80, 900),
                        created_at=datetime.now()
                    ),
                    Workflow(
                        name='图像处理工作流',
                        description='自动处理图像的工作流',
                        status='active',
                        total_executions=random.randint(50, 500),
                        successful_executions=random.randint(40, 450),
                        created_at=datetime.now()
                    ),
                ]
                session.add_all(workflows)
                session.commit()
                print(f"✓ 创建 {len(workflows)} 个工作流")
            else:
                print("ℹ️  工作流已存在")
        except Exception as e:
            session.rollback()
            print(f"⚠️  工作流创建跳过: {e}")
        
        # 4. 创建对话
        try:
            if session.query(Conversation).count() == 0:
                conversations = []
                for i in range(10):
                    conv = Conversation(
                        title=f'对话 {i+1}',
                        user_id=1,
                        model_id=random.choice([1, 2, 3, 4]),
                        status='active',
                        message_count=random.randint(5, 50),
                        created_at=datetime.now() - timedelta(days=random.randint(0, 30))
                    )
                    conversations.append(conv)
                session.add_all(conversations)
                session.commit()
                print(f"✓ 创建 {len(conversations)} 个对话")
            else:
                print("ℹ️  对话已存在")
        except Exception as e:
            session.rollback()
            print(f"⚠️  对话创建跳过: {e}")
        
        session.close()
        
        print("\n" + "=" * 60)
        print("数据库初始化完成！")
        print("=" * 60)
        print(f"\n数据库文件: {db_path}")
        print("\n您现在可以启动后端服务:")
        print("  python -m api.main")
        print("\n然后访问:")
        print("  Web界面: http://localhost:8000")
        print("  API文档: http://localhost:8000/docs")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("\n请确保已安装所有依赖:")
        print("  pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = init_database()
    sys.exit(0 if success else 1)
