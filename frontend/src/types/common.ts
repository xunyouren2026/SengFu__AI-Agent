// Base API response wrapper
export interface ApiResponse<T = unknown> {
  success: boolean;
  message?: string;
  timestamp?: string;
  request_id?: string;
  data?: T;
}

// Paginated response
export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

// Pagination params
export interface PaginationParams {
  page?: number;
  page_size?: number;
}

// Error response
export interface ErrorResponse {
  success: false;
  error_code: string;
  error_detail?: string;
  errors?: Record<string, unknown>[];
  message: string;
}

// Sort params
export interface SortParams {
  sort_field?: string;
  sort_order?: 'asc' | 'desc';
}

// Time period
export type TimePeriod = '1h' | '24h' | '7d' | '30d' | '90d';

// ID type
export type ID = string | number;

// Date string
export type DateTime = string;

// Generic entity with timestamps
export interface BaseEntity {
  id: string;
  created_at: DateTime;
  updated_at: DateTime;
}

// Select option
export interface SelectOption<T = string> {
  label: string;
  value: T;
  disabled?: boolean;
  icon?: string;
}

// Tab item
export interface TabItem {
  key: string;
  label: string;
  icon?: string;
  badge?: number | string;
  disabled?: boolean;
}

// Notification
export interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  read: boolean;
  created_at: DateTime;
  link?: string;
}

// Toast message
export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
}

// Chart data point
export interface ChartDataPoint {
  timestamp: DateTime;
  value: number;
  label?: string;
}

// Stats card
export interface StatsCard {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon?: string;
  color?: string;
}

// Breadcrumb item
export interface BreadcrumbItem {
  label: string;
  path?: string;
  icon?: string;
}

// Menu item for sidebar
export interface MenuItem {
  key: string;
  label: string;
  icon?: string;
  path?: string;
  badge?: number | string;
  children?: MenuItem[];
  disabled?: boolean;
  divider?: boolean;
}

// Menu group
export interface MenuGroup {
  label: string;
  icon?: string;
  items: MenuItem[];
}
