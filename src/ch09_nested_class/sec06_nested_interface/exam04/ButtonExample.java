package ch09_nested_class.sec06_nested_interface.exam04;
// 간결한 버전
public class ButtonExample {
	public static void main(String[] args) {
		clickButton("Ok 버튼을 클릭했습니다.");
		clickButton("Cancel 버튼을 클릭했습니다.");
	}
	static void clickButton(String msg) {
		Button btn = new Button(msg);

		//Ok 버튼 클릭 이벤트를 처리할 ClickListener 구현 클래스(로컬 클래스)
		// 단절시킴 즉 button.java 의 this.msg에서 받아와 처리
		class buttonListener implements Button.ClickListener {
			@Override
			public void onClick(String msg) {
				System.out.println(msg);
			}
		}

		//Ok 버튼 객체에 ClickListener 구현 객체 주입
		btn.setClickListener(new buttonListener());
		
		//Ok 버튼 클릭하기
		btn.click();
	}
}