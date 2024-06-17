package ch09_nested_class.sec06_nested_interface.exam03;
// 간결한 버전
public class ButtonExample3 {
	public static void main(String[] args) {
		clickButton("Ok 버튼을 클릭했습니다.");
		clickButton("Cancel 버튼을 클릭했습니다.");
	}
	static void clickButton(String msg) {
		Button btn = new Button();

		//Ok 버튼 클릭 이벤트를 처리할 ClickListener 구현 클래스(로컬 클래스)
		//버튼 클릭 이벤트를 처리하는 콜백 함수를 설정
		//비동기성: 콜백 함수는 특정 작업이 완료된 후에 실행되므로, 메인 프로그램의 흐름을 방해하지 않고 비동기적으로 동작
		//동작의 커스터마이즈: 특정 이벤트나 작업에 대해 동작을 유연하게 정의 -> 코드의 재사용성과 모듈화가 높아집니다.
		
		class OkListener implements Button.ClickListener {
			@Override
			public void onClick() {
				System.out.println(msg);
			}
		}

		//Ok 버튼 객체에 ClickListener 구현 객체 주입
		btn.setClickListener(new OkListener());
		
		//Ok 버튼 클릭하기
		btn.click();
	}
}