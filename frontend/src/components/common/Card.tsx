import { forwardRef } from 'react';
import { cn } from './Button';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'bordered' | 'elevated' | 'ghost';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hoverable?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(({ className, variant = 'default', padding = 'md', hoverable = false, children, ...props }, ref) => {
  const variants = {
    default: 'bg-white dark:bg-dark-800 rounded-xl shadow-sm',
    bordered: 'bg-white dark:bg-dark-800 rounded-xl border border-gray-200 dark:border-dark-600',
    elevated: 'bg-white dark:bg-dark-800 rounded-xl shadow-lg',
    ghost: 'bg-transparent rounded-xl',
  };
  const paddings = { none: '', sm: 'p-3', md: 'p-5', lg: 'p-8' };
  return (
    <div ref={ref} className={cn(variants[variant], paddings[padding], hoverable && 'hover:shadow-md transition-shadow cursor-pointer', className)} {...props}>
      {children}
    </div>
  );
});
Card.displayName = 'Card';

const CardHeader = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('flex items-center justify-between mb-4', className)} {...props} />
));
CardHeader.displayName = 'CardHeader';

const CardTitle = forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(({ className, ...props }, ref) => (
  <h3 ref={ref} className={cn('text-lg font-semibold text-gray-900 dark:text-white', className)} {...props} />
));
CardTitle.displayName = 'CardTitle';

const CardDescription = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(({ className, ...props }, ref) => (
  <p ref={ref} className={cn('text-sm text-gray-500 dark:text-gray-400', className)} {...props} />
));
CardDescription.displayName = 'CardDescription';

const CardContent = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('', className)} {...props} />
));
CardContent.displayName = 'CardContent';

const CardFooter = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('flex items-center justify-end gap-2 mt-4 pt-4 border-t border-gray-100 dark:border-dark-600', className)} {...props} />
));
CardFooter.displayName = 'CardFooter';

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter };
