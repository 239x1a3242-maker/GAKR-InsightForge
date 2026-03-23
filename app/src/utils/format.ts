export function formatDistanceToNow(date: string | Date): string {
  const now = new Date();
  const then = new Date(date);
  const diffInSeconds = Math.floor((now.getTime() - then.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return 'just now';
  }

  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return `${diffInMinutes} minute${diffInMinutes > 1 ? 's' : ''} ago`;
  }

  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`;
  }

  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 30) {
    return `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`;
  }

  const diffInMonths = Math.floor(diffInDays / 30);
  if (diffInMonths < 12) {
    return `${diffInMonths} month${diffInMonths > 1 ? 's' : ''} ago`;
  }

  const diffInYears = Math.floor(diffInMonths / 12);
  return `${diffInYears} year${diffInYears > 1 ? 's' : ''} ago`;
}

export function formatNumber(num: number, decimals: number = 2): string {
  if (num === null || num === undefined) return '-';
  
  if (Math.abs(num) >= 1e9) {
    return (num / 1e9).toFixed(decimals) + 'B';
  }
  if (Math.abs(num) >= 1e6) {
    return (num / 1e6).toFixed(decimals) + 'M';
  }
  if (Math.abs(num) >= 1e3) {
    return (num / 1e3).toFixed(decimals) + 'K';
  }
  
  return num.toFixed(decimals);
}

export function formatPercent(num: number, decimals: number = 1): string {
  if (num === null || num === undefined) return '-';
  return (num * 100).toFixed(decimals) + '%';
}

export function formatCurrency(num: number, currency: string = 'USD'): string {
  if (num === null || num === undefined) return '-';
  
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  });
  
  return formatter.format(num);
}

export function formatDate(date: string | Date): string {
  const d = new Date(date);
  return d.toLocaleDateString();
}

export function formatFileSize(bytes: number): string {
  if (bytes === null || bytes === undefined) return '-';
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  if (bytes === 0) return '0 B';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
}
