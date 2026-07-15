/**
 * API 响应解包辅助函数
 *
 * 背景：client.ts 的响应拦截器返回 response.data.data（后端 JSON 中 "data" 字段的值）。
 * 对于列表接口，后端 "data" 通常是数组，但前端页面期望 PaginatedResponse 格式：
 *   { data: [...], total, page, page_size, ... }
 * 拦截器丢弃了 total/page 等分页元数据，只保留了 data 数组。
 *
 * 此模块提供辅助函数，将拦截器解包后的原始数据包装为前端页面期望的格式。
 */

/**
 * 将拦截器解包后的列表响应包装为 { data: T[] } 格式。
 * - 如果 response 是数组，包装为 { data: response }
 * - 如果 response 已包含 data 字段（后端 data 字段本身是对象），直接返回
 * - 兜底：将整个对象包装为数组的唯一元素
 */
export function wrapListResponse<T>(response: any): { data: T[] } {
  if (Array.isArray(response)) {
    return { data: response };
  }
  if (response && typeof response === 'object' && 'data' in response) {
    return response;
  }
  return { data: Array.isArray(response) ? response : [response] };
}
