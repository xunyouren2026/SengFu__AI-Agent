#!/usr/bin/env python3
"""
修复数据库中的枚举值问题
将 'alibaba' 字符串更新为 'ALIBABA'
"""

import sqlite3
import os

def fix_model_provider_enum():
    """修复 ModelProvider 枚举值"""
    db_path = "./data/agi_framework.db"
    
    if not os.path.exists(db_path):
        print(f"数据库文件不存在: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 检查 models 表中是否有 provider 列
        cursor.execute("PRAGMA table_info(models)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'provider' in columns:
            # 更新小写的 'alibaba' 为大写的 'ALIBABA'
            cursor.execute("UPDATE models SET provider = 'ALIBABA' WHERE provider = 'alibaba'")
            updated = cursor.rowcount
            print(f"修复了 {updated} 条模型的 provider 字段")
        
        # 检查其他可能使用 ModelProvider 的表
        # 如果有其他表也使用这个枚举，需要一并修复
        
        conn.commit()
        print("数据库枚举值修复完成")
        
    except Exception as e:
        print(f"修复失败: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    fix_model_provider_enum()
