
package ch09_nested_class.sec06_nested_interface.exam04;

public class Button {
	private String msg;
	
	public Button(String msg) {
		this.msg = msg;
	}
	//정적 멤버 인터페이스
	public static interface ClickListener {
		//추상 메소드
		void onClick(String msg);
	}

	//필드
	private ClickListener clickListener;

	//메소드
	public void setClickListener(ClickListener clickListener) {
		this.clickListener = clickListener;
	}

	public void click() {
		this.clickListener.onClick(this.msg);
	}
}