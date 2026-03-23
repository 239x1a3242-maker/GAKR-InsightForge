/**
 * ResizableWidget — drag any corner/edge to resize a chart card.
 * Uses react-resizable under the hood.
 */
import { ResizableBox } from 'react-resizable';
import 'react-resizable/css/styles.css';
import type { ReactNode } from 'react';

// Column width unit in px (canvas is ~100% - sidebar, split into 5 cols)
export const COL_W = 220;   // min width per column unit
export const ROW_H = 280;   // default height

interface Props {
  children: ReactNode;
  defaultW?: number;   // px
  defaultH?: number;   // px
  minW?: number;
  minH?: number;
  maxW?: number;
  onResize?: (w: number, h: number) => void;
}

export function ResizableWidget({
  children,
  defaultW = COL_W * 2,
  defaultH = ROW_H,
  minW = COL_W,
  minH = 180,
  maxW = COL_W * 5,
  onResize,
}: Props) {
  return (
    <ResizableBox
      width={defaultW}
      height={defaultH}
      minConstraints={[minW, minH]}
      maxConstraints={[maxW, 800]}
      resizeHandles={['se', 'e', 's']}
      onResizeStop={(_, { size }) => onResize?.(size.width, size.height)}
      handle={
        <span
          className="react-resizable-handle react-resizable-handle-se"
          style={{
            position: 'absolute',
            right: 4,
            bottom: 4,
            width: 14,
            height: 14,
            cursor: 'se-resize',
            background: 'transparent',
            borderRight: '2px solid var(--accent)',
            borderBottom: '2px solid var(--accent)',
            borderRadius: '0 0 3px 0',
            opacity: 0.5,
            zIndex: 10,
          }}
        />
      }
      style={{ position: 'relative' }}
    >
      <div style={{ width: '100%', height: '100%' }}>
        {children}
      </div>
    </ResizableBox>
  );
}
